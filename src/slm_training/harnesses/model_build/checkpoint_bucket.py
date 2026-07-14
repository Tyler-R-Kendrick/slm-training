"""Push training checkpoints to a Hugging Face Bucket.

Default destination for real full training runs:
https://huggingface.co/buckets/TKendrick/OpenUI

Layout in the bucket::

    checkpoints/<run_id>/last.pt
    checkpoints/<run_id>/last.tokenizer.json
    checkpoints/<run_id>/last.meta.json
    checkpoints/<run_id>/best_ship_score.pt
    ...
    checkpoints/<run_id>/train_summary.json   # copied from the run dir when present
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT_BUCKET_ID = "TKendrick/OpenUI"
DEFAULT_CHECKPOINT_BUCKET_URI = f"hf://buckets/{DEFAULT_CHECKPOINT_BUCKET_ID}"

# Files we always try to sync from the local checkpoint dir.
_INCLUDE_GLOBS = (
    "*.pt",
    "*.tokenizer.json",
    "*.meta.json",
    "*.context.tokenizer.json",
    "promoted.json",
)


def normalize_bucket_uri(bucket: str | None) -> str:
    """Return ``hf://buckets/<id>(/prefix)`` for CLI / HfApi sync."""
    raw = (bucket or DEFAULT_CHECKPOINT_BUCKET_URI).strip()
    if not raw:
        raw = DEFAULT_CHECKPOINT_BUCKET_URI
    if raw.startswith("hf://buckets/"):
        return raw.rstrip("/")
    if raw.startswith("hf://"):
        raise ValueError(
            f"unsupported bucket URI {raw!r}; expected hf://buckets/<namespace>/<name>"
        )
    # Bare id or URL path.
    raw = raw.removeprefix("https://huggingface.co/buckets/").removeprefix(
        "http://huggingface.co/buckets/"
    )
    raw = raw.strip("/")
    if "/" not in raw:
        raise ValueError(
            f"bucket id {raw!r} must be namespace/name (e.g. {DEFAULT_CHECKPOINT_BUCKET_ID})"
        )
    return f"hf://buckets/{raw}"


def bucket_id_from_uri(uri: str) -> str:
    """Extract ``namespace/name`` from a bucket URI (ignore optional prefix)."""
    uri = normalize_bucket_uri(uri)
    rest = uri.removeprefix("hf://buckets/")
    parts = rest.split("/")
    if len(parts) < 2:
        raise ValueError(f"invalid bucket uri {uri!r}")
    return f"{parts[0]}/{parts[1]}"


def remote_run_prefix(uri: str, run_id: str) -> str:
    """Bucket URI prefix for one training run's checkpoints."""
    base = normalize_bucket_uri(uri)
    # If caller already appended a prefix beyond bucket id, keep it and add run_id.
    rest = base.removeprefix("hf://buckets/")
    parts = rest.split("/")
    bucket_root = f"hf://buckets/{parts[0]}/{parts[1]}"
    extra = "/".join(parts[2:]) if len(parts) > 2 else ""
    if extra:
        return f"{bucket_root}/{extra.rstrip('/')}/checkpoints/{run_id}"
    return f"{bucket_root}/checkpoints/{run_id}"


def resolve_sync_checkpoints(
    *,
    sync_checkpoints: bool | None,
    context_backend: str,
    explicit_bucket: str | None,
) -> bool:
    """Decide whether a train should push checkpoints.

    - ``False`` / unset: local-only (tests, matrix, programmatic harness calls).
    - ``True``: always sync (full HF CLI trains set this).
    - ``None`` (legacy auto): on only for HF-context when a bucket is selected
      and not disabled via env / empty bucket string.
    """
    if sync_checkpoints is False:
        return False
    if os.environ.get("SLM_DISABLE_CHECKPOINT_BUCKET", "").strip() in {
        "1",
        "true",
        "yes",
    }:
        return False
    if explicit_bucket is not None and explicit_bucket.strip() == "":
        return False
    if sync_checkpoints is True:
        return True
    # Legacy auto sentinel.
    if sync_checkpoints is None:
        return str(context_backend or "").lower() == "hf"
    return False


def _require_hub() -> Any:
    try:
        from huggingface_hub import HfApi, get_token  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required to sync checkpoints. "
            "Install with: pip install -e '.[hf]'"
        ) from exc
    return HfApi, get_token


def ensure_checkpoint_bucket(
    bucket: str | None = None,
    *,
    token: str | bool | None = None,
    private: bool = False,
) -> dict[str, Any]:
    """Create the bucket if missing (``exist_ok``). Requires write auth."""
    HfApi, get_token = _require_hub()
    uri = normalize_bucket_uri(bucket)
    bid = bucket_id_from_uri(uri)
    tok = token if token is not None else get_token()
    if not tok:
        raise RuntimeError(
            "Hugging Face auth required to create/ensure the checkpoint bucket. "
            "Set HF_TOKEN or run `hf auth login`."
        )
    api = HfApi(token=tok)
    url = api.create_bucket(bid, private=private, exist_ok=True, token=tok)
    info = api.bucket_info(bid, token=tok)
    return {
        "bucket_id": bid,
        "uri": uri,
        "url": f"https://huggingface.co/buckets/{bid}",
        "api_url": str(url),
        "private": bool(getattr(info, "private", private)),
    }


def _staging_dir(local_checkpoint_dir: Path, run_dir: Path | None) -> Path:
    """Build a clean staging folder containing only checkpoint artifacts (+ summary)."""
    stage = local_checkpoint_dir.parent / ".bucket_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for pattern in _INCLUDE_GLOBS:
        for path in sorted(local_checkpoint_dir.glob(pattern)):
            if not path.is_file():
                continue
            dest = stage / path.name
            shutil.copy2(path, dest)
            copied.append(path.name)
    if run_dir is not None:
        summary = run_dir / "train_summary.json"
        if summary.is_file():
            shutil.copy2(summary, stage / "train_summary.json")
            copied.append("train_summary.json")
    if not copied:
        raise FileNotFoundError(
            f"no checkpoint artifacts to sync under {local_checkpoint_dir}"
        )
    return stage


def sync_run_checkpoints(
    local_checkpoint_dir: Path | str,
    *,
    run_id: str,
    bucket: str | None = None,
    run_dir: Path | str | None = None,
    token: str | bool | None = None,
    dry_run: bool = False,
    ensure_bucket: bool = True,
) -> dict[str, Any]:
    """Upload one run's checkpoint directory to the HF bucket.

    Returns a JSON-serializable report with remote URI and sync plan summary.
    """
    HfApi, get_token = _require_hub()
    local_checkpoint_dir = Path(local_checkpoint_dir)
    run_dir_path = Path(run_dir) if run_dir is not None else local_checkpoint_dir.parent
    uri = normalize_bucket_uri(bucket)
    remote = remote_run_prefix(uri, run_id)
    tok = token if token is not None else get_token()
    if not tok and not dry_run:
        raise RuntimeError(
            "Full training checkpoint sync requires Hugging Face auth. "
            "Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) or run `hf auth login`. "
            "For local-only / CI scratch runs use --no-sync-checkpoints."
        )

    if ensure_bucket and not dry_run:
        ensure_checkpoint_bucket(uri, token=tok)

    stage = _staging_dir(local_checkpoint_dir, run_dir_path)
    api = HfApi(token=tok)
    plan = api.sync_bucket(
        source=str(stage),
        dest=remote,
        dry_run=dry_run,
        token=tok,
    )
    # sync_bucket returns a SyncPlan; summarize defensively.
    summary = {
        "ok": True,
        "dry_run": bool(dry_run),
        "run_id": run_id,
        "local_checkpoint_dir": str(local_checkpoint_dir.as_posix()),
        "remote_uri": remote,
        "bucket_url": f"https://huggingface.co/buckets/{bucket_id_from_uri(uri)}",
        "files": sorted(p.name for p in stage.iterdir() if p.is_file()),
        "plan": _plan_to_dict(plan),
    }
    # Best-effort cleanup of staging (keep on dry-run failures / debug).
    if not dry_run:
        shutil.rmtree(stage, ignore_errors=True)
    return summary


def _plan_to_dict(plan: Any) -> dict[str, Any]:
    if plan is None:
        return {}
    if isinstance(plan, dict):
        return plan
    out: dict[str, Any] = {}
    for key in (
        "uploads",
        "downloads",
        "deletes",
        "skips",
        "total_size",
        "operations",
        "summary",
    ):
        if hasattr(plan, key):
            val = getattr(plan, key)
            if key == "operations" and val is not None:
                try:
                    out[key] = [
                        {
                            "action": getattr(op, "action", None),
                            "path": getattr(op, "path", None),
                            "size": getattr(op, "size", None),
                        }
                        for op in list(val)[:50]
                    ]
                except TypeError:
                    out[key] = str(val)
            else:
                out[key] = val
    if not out:
        out["repr"] = repr(plan)
    return out


def maybe_sync_train_checkpoints(config: Any, checkpoint_dir: Path) -> dict[str, Any] | None:
    """Hook used by ``train()`` — returns sync report or None when disabled."""
    enabled = resolve_sync_checkpoints(
        sync_checkpoints=getattr(config, "sync_checkpoints", None),
        context_backend=str(getattr(config, "context_backend", "") or ""),
        explicit_bucket=getattr(config, "checkpoint_bucket", None),
    )
    if not enabled:
        return None
    bucket = getattr(config, "checkpoint_bucket", None) or os.environ.get(
        "SLM_CHECKPOINT_BUCKET", DEFAULT_CHECKPOINT_BUCKET_URI
    )
    return sync_run_checkpoints(
        checkpoint_dir,
        run_id=str(config.run_id),
        bucket=bucket,
        run_dir=Path(config.run_dir),
        dry_run=bool(getattr(config, "checkpoint_bucket_dry_run", False)),
        ensure_bucket=True,
    )
