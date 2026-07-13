"""Migrate TwoTower checkpoints across tokenizer vocabulary changes."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.models.tokenizer import OpenUITokenizer, TOKENIZER_VERSION
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


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

    payload = torch.load(source_checkpoint, map_location=device, weights_only=False)
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
    valid = {f.name for f in TwoTowerConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    cfg = TwoTowerConfig(**{k: v for k, v in raw_cfg.items() if k in valid})

    max_prompt = max(
        (len(new_tokenizer.encode(r.prompt)) for r in records), default=16
    )
    max_target = max(
        (len(new_tokenizer.encode(r.openui)) for r in records), default=32
    )
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
            module = key.split(".tok.weight")[0]
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
