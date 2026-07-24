"""Explicit migrations for TwoTower vocab and grammar topology checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from slm_training.dsl.schema import load_jsonl
from slm_training.models.tokenizer import (
    OpenUITokenizer,
    TOKENIZER_VERSION,
    load_tokenizer_sidecar,
)
from slm_training.models.twotower import (
    TwoTowerConfig,
    TwoTowerModel,
    migrate_recursive_depth_aux_config,
)


def _copy_shared_embedding_rows(
    *,
    old_weight: torch.Tensor,
    old_token_to_id: dict[str, int],
    new_weight: torch.Tensor,
    new_token_to_id: dict[str, int],
) -> int:
    """Copy rows for token strings present in both vocabs. Returns copied count."""
    copied = 0
    with torch.no_grad():
        for token, new_id in new_token_to_id.items():
            old_id = old_token_to_id.get(token)
            if old_id is None:
                continue
            if old_id >= old_weight.shape[0] or new_id >= new_weight.shape[0]:
                continue
            new_weight[new_id] = old_weight[old_id]
            copied += 1
    return copied


def migrate_twotower_checkpoint(
    *,
    source_checkpoint: Path | str,
    train_records_path: Path | str,
    output_checkpoint: Path | str,
    device: str = "cpu",
) -> dict:
    """
    Rebuild a v2 tokenizer from train records and remap embedding weights.

    Transformer blocks (context + denoiser layers, norms, positions) are copied
    when tensor shapes match. Token embeddings are remapped by shared token
    string; new subtoken rows stay randomly initialized.
    """
    source_checkpoint = Path(source_checkpoint)
    output_checkpoint = Path(output_checkpoint)
    train_records_path = Path(train_records_path)

    payload = torch.load(source_checkpoint, map_location=device, weights_only=True)
    if payload.get("kind") != "twotower":
        raise ValueError(f"checkpoint kind {payload.get('kind')!r} is not twotower")

    tok_path = source_checkpoint.with_suffix(".tokenizer.json")
    if not tok_path.exists():
        raise FileNotFoundError(f"missing tokenizer next to checkpoint: {tok_path}")

    old_tokenizer = OpenUITokenizer.load(tok_path, allow_legacy=True)
    records = load_jsonl(train_records_path)
    texts = [r.prompt for r in records] + [r.openui for r in records]
    new_tokenizer = OpenUITokenizer.build(texts)

    raw_cfg = dict(payload.get("config") or {})
    if isinstance(raw_cfg.get("grammar_ltr_stages"), list):
        raw_cfg["grammar_ltr_stages"] = tuple(raw_cfg["grammar_ltr_stages"])
    raw_cfg = migrate_recursive_depth_aux_config(raw_cfg)
    valid = {f.name for f in TwoTowerConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    cfg = TwoTowerConfig(**{k: v for k, v in raw_cfg.items() if k in valid})

    max_prompt = max((len(new_tokenizer.encode(r.prompt)) for r in records), default=16)
    max_target = max((len(new_tokenizer.encode(r.openui)) for r in records), default=32)
    cfg.max_prompt_len = max(cfg.max_prompt_len, max_prompt + 4)
    cfg.max_target_len = max(cfg.max_target_len, max_target + 8)

    new_model = TwoTowerModel(tokenizer=new_tokenizer, config=cfg, device=device)
    if "gen_len" in payload:
        new_model.gen_len = int(payload["gen_len"])
    else:
        new_model.gen_len = max(max_target + 2, 16)

    old_state = payload["state_dict"]
    new_state = new_model.state_dict()
    copied_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, new_tensor in new_state.items():
        old_tensor = old_state.get(key)
        if old_tensor is None:
            skipped_keys.append(key)
            continue
        if key.endswith(".tok.weight"):
            copied = _copy_shared_embedding_rows(
                old_weight=old_tensor,
                old_token_to_id=old_tokenizer.token_to_id,
                new_weight=new_tensor,
                new_token_to_id=new_tokenizer.token_to_id,
            )
            new_state[key] = new_tensor
            copied_keys.append(f"{key}({copied} rows)")
            continue
        if old_tensor.shape == new_tensor.shape:
            new_state[key] = old_tensor
            copied_keys.append(key)
        else:
            skipped_keys.append(key)

    new_model.load_state_dict(new_state, strict=True)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    new_model.save(output_checkpoint)

    report = {
        "source_checkpoint": str(source_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "old_tokenizer_version": old_tokenizer.version,
        "new_tokenizer_version": TOKENIZER_VERSION,
        "old_vocab_size": old_tokenizer.vocab_size,
        "new_vocab_size": new_tokenizer.vocab_size,
        "shared_token_rows_copied": sum(
            int(part.split("(")[1].rstrip(" rows)"))
            for part in copied_keys
            if ".tok.weight(" in part
        ),
        "copied_keys": copied_keys,
        "skipped_keys": skipped_keys,
        "train_records": str(train_records_path),
    }
    report_path = output_checkpoint.with_suffix(".migrate.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def migrate_grammar_diffusion_checkpoint(
    *,
    source_checkpoint: Path | str,
    output_checkpoint: Path | str,
    device: str = "cpu",
) -> dict:
    """Warm-start topology checkpoint v2 from a fixed-canvas grammar checkpoint."""
    from slm_training.models.grammar_diffusion import (
        GrammarDiffusionConfig,
        GrammarDiffusionModel,
        _restore_codec,
    )

    source_checkpoint = Path(source_checkpoint)
    output_checkpoint = Path(output_checkpoint)
    payload = torch.load(source_checkpoint, map_location=device, weights_only=True)
    if payload.get("kind") != "grammar_diffusion":
        raise ValueError(
            f"checkpoint kind {payload.get('kind')!r} is not grammar_diffusion"
        )
    if (
        int(payload.get("format_version") or 1)
        >= GrammarDiffusionModel.CHECKPOINT_FORMAT
    ):
        raise ValueError("grammar checkpoint is already topology format v2")
    tokenizer_path = source_checkpoint.with_suffix(".tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"missing tokenizer next to checkpoint: {tokenizer_path}"
        )
    tokenizer = OpenUITokenizer.load(tokenizer_path, allow_legacy=True)
    codec = _restore_codec(payload.get("codec") or {})
    raw_config = dict(payload.get("config") or {})
    valid = set(GrammarDiffusionConfig.__dataclass_fields__)
    config = GrammarDiffusionConfig(
        **{key: value for key, value in raw_config.items() if key in valid}
    )
    model = GrammarDiffusionModel(tokenizer, codec, config, device)
    old_state = payload.get("state_dict") or {}
    new_state = model.state_dict()
    copied_keys: list[str] = []
    skipped_old_keys: list[str] = []
    initialized_keys: list[str] = []
    for key, tensor in old_state.items():
        if key in new_state and new_state[key].shape == tensor.shape:
            new_state[key] = tensor
            copied_keys.append(key)
        else:
            skipped_old_keys.append(key)
    for key in new_state:
        if key not in copied_keys:
            initialized_keys.append(key)
    model.load_state_dict(new_state, strict=True)
    model.save(output_checkpoint)
    report = {
        "source_checkpoint": str(source_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "source_format_version": int(payload.get("format_version") or 1),
        "output_format_version": GrammarDiffusionModel.CHECKPOINT_FORMAT,
        "warm_start_only": True,
        "copied_keys": copied_keys,
        "skipped_old_keys": skipped_old_keys,
        "initialized_keys": initialized_keys,
    }
    output_checkpoint.with_suffix(".migrate.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def migrate_to_shared_recursive_denoiser(
    old_path: Path | str,
    new_path: Path | str,
    config: dict[str, object] | None = None,
    device: str = "cpu",
) -> dict:
    """Warm-start a stacked DenoiserTower checkpoint into a shared-recursive tower.

    Loads the source checkpoint, builds a ``TwoTowerModel`` with
    ``denoiser_arch="shared_recursive"``, copies all matching ``denoiser.*``
    keys, leaves the new z-state tensors (``z_latent``, ``ctx_proj``) randomly
    initialized, and writes the result to ``new_path``.
    """
    old_path = Path(old_path)
    new_path = Path(new_path)
    payload = torch.load(old_path, map_location=device, weights_only=True)
    if payload.get("kind") != "twotower":
        raise ValueError(f"checkpoint kind {payload.get('kind')!r} is not twotower")

    tokenizer_path = old_path.with_suffix(".tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"missing tokenizer next to checkpoint: {tokenizer_path}")
    tokenizer = load_tokenizer_sidecar(tokenizer_path, allow_legacy=True)

    raw_cfg = dict(payload.get("config") or {})
    if isinstance(raw_cfg.get("grammar_ltr_stages"), list):
        raw_cfg["grammar_ltr_stages"] = tuple(raw_cfg["grammar_ltr_stages"])
    raw_cfg = migrate_recursive_depth_aux_config(raw_cfg)
    valid = {f.name for f in TwoTowerConfig.__dataclass_fields__.values()}
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k in valid}
    cfg_kwargs["denoiser_arch"] = "shared_recursive"
    user_cfg = config or {}
    for key in (
        "recursive_steps",
        "recursive_transition_layers",
        "recursive_depth_supervision_weights",
        "recursive_depth_aux_mode",
        "recursive_depth_aux_weight",
        "recursive_update_mode",
        "recursive_empty_f_mode",
        "recursive_norm_mode",
    ):
        if key in user_cfg:
            cfg_kwargs[key] = user_cfg[key]
    cfg = TwoTowerConfig(**cfg_kwargs)

    new_model = TwoTowerModel(tokenizer=tokenizer, config=cfg, device=device)
    old_state = payload["state_dict"]
    new_state = new_model.state_dict()
    copied_keys: list[str] = []
    skipped_keys: list[str] = []
    initialized_keys: list[str] = []

    for key, tensor in new_state.items():
        if key in old_state and old_state[key].shape == tensor.shape:
            new_state[key] = old_state[key]
            copied_keys.append(key)
        else:
            initialized_keys.append(key)
    for key in old_state:
        if key not in copied_keys:
            skipped_keys.append(key)

    new_model.load_state_dict(new_state, strict=False)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_model.save(new_path)

    report = {
        "source_checkpoint": str(old_path),
        "output_checkpoint": str(new_path),
        "denoiser_arch": "shared_recursive",
        "recursive_steps": cfg.recursive_steps,
        "recursive_transition_layers": int(
            cfg.recursive_transition_layers or cfg.denoiser_layers
        ),
        "copied_keys": copied_keys,
        "skipped_old_keys": skipped_keys,
        "initialized_keys": initialized_keys,
    }
    report_path = new_path.with_suffix(".migrate.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
