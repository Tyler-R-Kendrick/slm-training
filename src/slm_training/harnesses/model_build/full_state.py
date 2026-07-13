"""Full training-state checkpoints for bit-exact resume.

A serving checkpoint (``last.pt``) only carries model weights + config. A
hill-climbing loop additionally needs a checkpoint that can reproduce the
*next optimizer update* after a restart: optimizer moments, grad-scaler state,
every RNG stream (loop shuffle, model corruption, torch, CUDA), the pending
batch queue, token counters, and the identity of the data + code that
produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FULL_STATE_VERSION = 1


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:  # noqa: BLE001
        return None


def data_manifest_sha(train_dir: Path) -> str | None:
    """Identity of the training corpus: manifest content fingerprint or file hash."""
    train_dir = Path(train_dir)
    manifest = train_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            fp = data.get("content_fingerprint")
            if fp:
                return str(fp)
        except Exception:  # noqa: BLE001
            pass
    records = train_dir / "records.jsonl"
    if records.exists():
        h = hashlib.sha256()
        h.update(records.read_bytes())
        return h.hexdigest()
    return None


def _jsonable_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config) and not isinstance(config, type):
        raw = asdict(config)
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raw = {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def save_full_state(
    path: Path | str,
    *,
    plugin: Any,
    optimizer: Any,
    scaler: Any,
    step: int,
    seen_prompt_tokens: int,
    seen_target_tokens: int,
    loop_rng: Any,
    pending_batches: list[list[Any]],
    config: Any,
    manifest_sha: str | None,
    best_weighted_nll: float | None = None,
    best_ship_score: float | None = None,
    mixture_hash: str | None = None,
) -> Path:
    """Atomically write a resumable training-state checkpoint."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(plugin, "_state_dict_for_checkpoint"):
        model_state = plugin._state_dict_for_checkpoint()
    elif hasattr(plugin, "state_dict"):
        model_state = {k: v.cpu() for k, v in plugin.state_dict().items()}
    else:
        model_state = None

    payload = {
        "kind": "full_train_state",
        "version": FULL_STATE_VERSION,
        "step": int(step),
        "seen_prompt_tokens": int(seen_prompt_tokens),
        "seen_target_tokens": int(seen_target_tokens),
        "model": model_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        # accel returns a stateless _NullScaler when AMP is off.
        "scaler": (
            scaler.state_dict()
            if scaler is not None and hasattr(scaler, "state_dict")
            else None
        ),
        "loop_rng": loop_rng.getstate() if loop_rng is not None else None,
        "model_mask_rng": (
            plugin._rng.getstate() if hasattr(plugin, "_rng") else None
        ),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "pending_batch_ids": [
            [r.id for r in batch] for batch in (pending_batches or [])
        ],
        "data_manifest_sha": manifest_sha,
        "mixture_hash": mixture_hash,
        "code_git_sha": _git_sha(),
        "config": _jsonable_config(config),
        "best_weighted_nll": best_weighted_nll,
        "best_ship_score": best_ship_score,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return path


def load_full_state(path: Path | str) -> dict[str, Any]:
    """Load a full-state checkpoint (trusted artifact — carries RNG tuples)."""
    import torch

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != "full_train_state":
        raise ValueError(f"{path} is not a full_train_state checkpoint")
    return payload


def restore_rng_states(payload: dict[str, Any], *, plugin: Any, loop_rng: Any) -> None:
    """Restore every RNG stream captured by :func:`save_full_state`."""
    import torch

    if payload.get("loop_rng") is not None and loop_rng is not None:
        loop_rng.setstate(payload["loop_rng"])
    if payload.get("model_mask_rng") is not None and hasattr(plugin, "_rng"):
        plugin._rng.setstate(payload["model_mask_rng"])
    if payload.get("torch_rng") is not None:
        torch.set_rng_state(payload["torch_rng"])
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        except Exception:  # noqa: BLE001
            pass
