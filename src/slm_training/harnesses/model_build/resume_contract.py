"""Exact-continuation compatibility and exposure units for the existing trainer."""

from __future__ import annotations

import hashlib
import json
import platform
import inspect
from dataclasses import asdict
from pathlib import Path


# Invocation/reporting locations and a later total stopping cursor may differ.
# Every other resolved field (including seed, data, objective and cadence) matches.
_INVOCATION_FIELDS = frozenset(
    {
        "run_root",
        "run_id",
        "resume_from",
        "steps",
        "max_wall_minutes",
        "checkpoint_every_steps",
        "telemetry",
        "telemetry_sample_interval_ms",
        "checkpoint_bucket",
        "checkpoint_bucket_dry_run",
        "sync_checkpoints",
    }
)


def resume_identity(config, plugin) -> dict:
    import torch

    recipe = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in asdict(config).items()
        if k not in _INVOCATION_FIELDS
    }
    tokenizers = {}
    for name in ("tokenizer", "context_tokenizer"):
        tokenizer = getattr(plugin, name, None)
        if tokenizer is not None:
            tokenizers[name] = {
                "type": type(tokenizer).__qualname__,
                "token_to_id": getattr(tokenizer, "token_to_id", None),
                "layout": getattr(tokenizer, "layout_fingerprint", None),
            }
    source = {}
    for module in (
        Path(__file__),
        Path(__file__).with_name("full_state.py"),
        Path(__file__).with_name("train_loop.py"),
        Path(inspect.getfile(type(plugin))),
    ):
        source[module.name] = hashlib.sha256(module.read_bytes()).hexdigest()
    return {
        "recipe": recipe,
        "tokenizers": tokenizers,
        "source": source,
        "data_content": data_content_identity(config.train_dir),
        "replay_content": data_content_identity(config.replay_train_dir)
        if config.replay_train_dir
        else None,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "machine": platform.machine(),
            "threads": torch.get_num_threads(),
            "device": config.device,
            "cuda": torch.version.cuda,
        },
    }


def validate_resume_contract(
    payload: dict, config, plugin, manifest_sha: str | None, optimizer=None, scaler=None
) -> dict | None:
    if payload.get("version") != 2 or payload.get("resume_contract") is None:
        raise ValueError(
            "resume_from legacy state lacks an exact-continuation contract; use weight initialization"
        )
    if not manifest_sha or payload.get("data_manifest_sha") != manifest_sha:
        raise ValueError("resume_from data mismatch")
    compatibility = _match_resume_identity(
        payload["resume_contract"], resume_identity(config, plugin)
    )
    if payload.get("optimizer_fingerprint") != getattr(optimizer, "fingerprint", None):
        raise ValueError("optimizer fingerprint mismatch")
    required = (
        "model",
        "optimizer",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "loop_rng",
        "pending_batch_ids",
        "snapshot_tokens",
    )
    if any(payload.get(key) is None for key in required):
        raise ValueError("resume_from missing exact state")
    _validate_optional_streams(payload, config, plugin, scaler)
    if payload.get("accumulation_position") != 0:
        raise ValueError("resume_from uncommitted gradients")
    for key in (
        "step",
        "seen_prompt_tokens",
        "seen_target_tokens",
        "seen_primary_examples",
        "seen_replay_examples",
    ):
        if type(payload.get(key)) is not int or payload[key] < 0:
            raise ValueError(f"resume_from invalid counter:{key}")
    if config.steps < payload["step"]:
        raise ValueError("resume_from requested total steps precede cursor")
    return compatibility


def _approved_source_migration(previous: dict, current: dict) -> dict | None:
    if previous == current:
        return None
    path = (
        Path(__file__).parents[2] / "resources/model_build/resume_compatibility_v1.json"
    )
    raw = path.read_bytes()
    migration = json.loads(raw)
    if (
        set(migration) != {"schema_version", "from_sources", "to_source", "reason"}
        or migration["schema_version"] != "resume_source_migration/v2"
        or not isinstance(migration["from_sources"], list)
        or not migration["from_sources"]
        or set(migration["to_source"]) != {"resume_contract.py", "train_loop.py"}
        or any(
            set(source) != set(migration["to_source"])
            for source in migration["from_sources"]
        )
    ):
        raise ValueError("invalid controller-owned resume migration")
    if not any(
        all(previous.get(k) == v for k, v in source.items())
        for source in migration["from_sources"]
    ):
        raise ValueError("resume_from recipe/tokenizer/source/runtime mismatch")
    if {**previous, **migration["to_source"]} != current:
        raise ValueError("resume_from recipe/tokenizer/source/runtime mismatch")
    return {
        "schema_version": migration["schema_version"],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "from_source": previous,
        "to_source": current,
    }


def _match_resume_identity(previous: dict, current: dict) -> dict | None:
    """An initialization path is provenance, not a repeated initialization.

    No optimizer reset, new ancestor or recipe change is accepted. Old source
    is usable only through the separately pinned, controller-owned migration.
    Neither the original full-state file nor its recorded recipe is modified.
    """
    recipe = dict(previous["recipe"])
    initializer = recipe.get("initialize_from")
    if initializer and current["recipe"].get("initialize_from") is None:
        if recipe.get("initialization_weight_retention", 0):
            raise ValueError("resume_from missing initialized-weight anchor state")
        recipe["initialize_from"] = None
    migration = _approved_source_migration(previous["source"], current["source"])
    normalized = {**previous, "recipe": recipe, "source": current["source"]}
    if normalized != current:
        raise ValueError("resume_from recipe/tokenizer/source/runtime mismatch")
    if initializer or migration:
        return {"original_initializer": initializer, "source_migration": migration}
    return None


def _validate_optional_streams(payload, config, plugin, scaler) -> None:
    # Absence must be explicit; missing state is not an unused feature.
    if not {"scaler", "scheduler", "cuda_rng", "model_mask_rng"} <= payload.keys():
        raise ValueError("resume_from missing exact state")
    if payload["scheduler"] is not None:
        raise ValueError("resume_from unsupported scheduler")
    if hasattr(plugin, "_rng") and payload["model_mask_rng"] is None:
        raise ValueError("resume_from missing model masking RNG")
    if callable(getattr(scaler, "state_dict", None)) and payload["scaler"] is None:
        raise ValueError("resume_from missing AMP scaler")
    if str(config.device).startswith("cuda") and payload["cuda_rng"] is None:
        raise ValueError("resume_from missing CUDA RNG")


def exposure_by_snapshot(
    primary_sha,
    replay_sha,
    primary_count,
    replay_count,
    primary_examples,
    replay_examples,
    snapshot_tokens=None,
) -> list[dict]:
    """Consumed examples divided by their own immutable snapshot size.

    Counters come from actual batches, already include accumulation, and are
    local-worker counts. This trainer does not implement distributed sampling.
    """
    rows = []
    for role, digest, count, examples in (
        ("primary", primary_sha, primary_count, primary_examples),
        ("replay", replay_sha, replay_count, replay_examples),
    ):
        if not count and not examples:
            continue
        if (
            type(count) is not int
            or count <= 0
            or type(examples) is not int
            or examples < 0
        ):
            raise ValueError("exposure:invalid_count")
        if not digest:
            raise ValueError("exposure:missing_snapshot")
        rows.append(
            {
                "snapshot_sha256": digest,
                "record_count": count,
                "consumed_examples": examples,
                "example_exposure_epochs": examples / count,
                "tokens": (snapshot_tokens or {}).get(role),
            }
        )
    return rows


def count_snapshot_tokens(plugin, batch, counters) -> tuple[int, int]:
    """Count actual unpadded tokens by primary/replay source, once per batch."""
    total_prompt = total_target = 0
    for role, replay in (("primary", False), ("replay", True)):
        subset = [
            record for record in batch if record.id.startswith("replay::") == replay
        ]
        if not subset:
            continue
        prompt, target = plugin.count_batch_tokens(subset)
        current = counters.setdefault(role, {"prompt": 0, "target": 0})
        current["prompt"] += int(prompt)
        current["target"] += int(target)
        total_prompt += int(prompt)
        total_target += int(target)
    return total_prompt, total_target


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def data_content_identity(train_dir: Path) -> str | None:
    """Bind actual records and manifest; a stale manifest is not data identity."""
    records = Path(train_dir) / "records.jsonl"
    if not records.is_file():
        return None
    manifest = Path(train_dir) / "manifest.json"
    payload = {
        "records": file_sha256(records),
        "manifest": file_sha256(manifest) if manifest.is_file() else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
