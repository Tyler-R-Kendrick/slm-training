"""Resolve which checkpoint the web playground should serve.

"Serve the latest model we're building", in priority order:

1. an explicit path handed by the caller (CLI flag / ``create_app`` argument)
2. the ``SLM_PLAYGROUND_CHECKPOINT`` environment pin
3. the atomically deployed lineage pointer (``outputs/lineage/deployments``)
4. the promoted lineage champion (``outputs/lineage/champions``)
5. the newest loadable training-run checkpoint under ``outputs/``
6. the committed playground demo fixture

A candidate only wins if it is actually loadable by the serving stack: the
checkpoint file, its ``.tokenizer.json`` and ``.meta.json`` sidecars must
exist, the recorded kind must be ``twotower``, and — when the caller can only
run ONNX inference — the exported ``.context.onnx`` / ``.denoiser.onnx``
sidecars must exist too. Remote ``artifact_uri`` values (``hf://…``) are
skipped: this resolver never downloads.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT

ENV_CHECKPOINT = "SLM_PLAYGROUND_CHECKPOINT"

# outputs/ layouts that hold run checkpoints (see docs/MODEL_CARD.md roster).
_RUN_GLOBS = (
    "outputs/runs/**/checkpoints/last.pt",
    "outputs/autoresearch/**/checkpoints/last.pt",
)
_LINEAGE_TRACKS = ("twotower", "causal_lm")


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """A serving decision: which checkpoint, and why it was picked."""

    path: Path
    provenance: str  # explicit | env | deployment | champion | latest-run | demo-fixture
    run_id: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "path": str(self.path),
            "provenance": self.provenance,
            "run_id": self.run_id,
            "detail": self.detail,
        }


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def checkpoint_is_loadable(path: Path, *, require_onnx: bool = False) -> bool:
    """True when the serving stack can actually load ``path``."""
    if not path.is_file():
        return False
    if not path.with_suffix(".tokenizer.json").is_file():
        return False
    meta = _read_json(path.with_suffix(".meta.json"))
    if meta is None or str(meta.get("kind") or "") != "twotower":
        return False
    if require_onnx:
        stem = path.with_suffix("")
        for sidecar in (".context.onnx", ".denoiser.onnx"):
            if not stem.with_suffix(sidecar).is_file():
                return False
    return True


def _run_id_for(path: Path) -> str:
    """Human-meaningful run id for a checkpoint path (…/<run>/checkpoints/last.pt)."""
    parts = path.parts
    if "checkpoints" in parts:
        index = parts.index("checkpoints")
        if index > 0:
            return parts[index - 1]
    return path.parent.name or path.stem


def _pointer_candidates(root: Path) -> Iterator[tuple[str, dict]]:
    """Yield (provenance, pointer-record) pairs, deployments before champions."""
    deployments = root / "outputs" / "lineage" / "deployments"
    selected = _read_json(deployments / "selected.json")
    if selected and selected.get("record"):
        record = _read_json(root / "outputs" / "lineage" / str(selected["record"]))
        if record:
            yield "deployment", record
    for track in _LINEAGE_TRACKS:
        current = _read_json(deployments / track / "current.json")
        if current and current.get("record"):
            record = _read_json(deployments / track / str(current["record"]))
            if record:
                yield "deployment", record
    champions = root / "outputs" / "lineage" / "champions"
    for track in _LINEAGE_TRACKS:
        current = _read_json(champions / track / "current.json")
        if current and current.get("record"):
            record = _read_json(champions / track / str(current["record"]))
            if record:
                yield "champion", record


def _pointer_path(root: Path, record: dict) -> Path | None:
    uri = str(record.get("artifact_uri") or "").strip()
    if not uri or "://" in uri:
        return None  # remote artifact; this resolver never downloads
    path = Path(uri)
    return path if path.is_absolute() else root / path


def _latest_run_checkpoint(root: Path, *, require_onnx: bool) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for pattern in _RUN_GLOBS:
        for path in root.glob(pattern):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, path))
    for _, path in sorted(candidates, key=lambda item: item[0], reverse=True):
        if checkpoint_is_loadable(path, require_onnx=require_onnx):
            return path
    return None


def resolve_serving_checkpoint(
    *,
    explicit: Path | str | None = None,
    root: Path | str | None = None,
    require_onnx: bool = False,
) -> ResolvedCheckpoint:
    """Pick the checkpoint the playground should serve right now."""
    if explicit is not None:
        path = Path(explicit)
        return ResolvedCheckpoint(
            path=path, provenance="explicit", run_id=_run_id_for(path)
        )
    base = Path(root) if root is not None else Path(".")

    env_pin = (os.getenv(ENV_CHECKPOINT) or "").strip()
    if env_pin:
        path = Path(env_pin)
        if not path.is_absolute():
            path = base / path
        return ResolvedCheckpoint(
            path=path, provenance="env", run_id=_run_id_for(path)
        )

    for provenance, record in _pointer_candidates(base):
        path = _pointer_path(base, record)
        if path is not None and checkpoint_is_loadable(path, require_onnx=require_onnx):
            return ResolvedCheckpoint(
                path=path,
                provenance=provenance,
                run_id=str(record.get("run_id") or _run_id_for(path)),
                detail=str(record.get("pointer_id") or "") or None,
            )

    latest = _latest_run_checkpoint(base, require_onnx=require_onnx)
    if latest is not None:
        return ResolvedCheckpoint(
            path=latest, provenance="latest-run", run_id=_run_id_for(latest)
        )

    demo = PLAYGROUND_DEMO_CHECKPOINT
    if not demo.is_absolute() and root is not None:
        anchored = base / demo
        if anchored.exists():
            demo = anchored
    return ResolvedCheckpoint(
        path=demo,
        provenance="demo-fixture",
        run_id="playground_demo",
        detail="no deployed, champion, or run checkpoint found under outputs/",
    )
