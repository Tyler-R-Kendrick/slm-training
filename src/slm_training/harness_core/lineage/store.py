"""Filesystem-backed immutable lineage store with atomic mutable pointers."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from slm_training.harness_core.lineage.records import (
    ChampionPointer,
    DataSnapshot,
    EvaluationReport,
    MergeManifest,
    RunManifest,
)
from slm_training.harness_core.lineage.records import content_sha

Record = TypeVar(
    "Record",
    RunManifest,
    DataSnapshot,
    EvaluationReport,
    MergeManifest,
    ChampionPointer,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_new(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(
            f"immutable lineage record already exists: {path}"
        ) from exc
    return path


def _atomic_write(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_from_dict(data: dict[str, Any]) -> RunManifest:
    for key in ("parent_ids", "artifact_uris"):
        data[key] = tuple(data.get(key) or ())
    return RunManifest(**data)


class LineageStore:
    def __init__(self, root: Path | str = Path("outputs/lineage")) -> None:
        self.root = Path(root)

    def create_run(self, manifest: RunManifest) -> Path:
        run_dir = self.root / "runs" / manifest.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_new(run_dir / "manifest.json", manifest.to_dict())
        _atomic_write(run_dir / "current.json", {"record": "manifest.json"})
        return run_dir

    def load_run(self, run_id: str) -> RunManifest:
        run_dir = self.root / "runs" / run_id
        current = run_dir / "current.json"
        record = _read(current)["record"] if current.exists() else "manifest.json"
        return _run_from_dict(_read(run_dir / record))

    def transition_run(
        self,
        run_id: str,
        state: str,
        *,
        artifact_uris: tuple[str, ...] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> RunManifest:
        current = self.load_run(run_id)
        allowed = {
            "running": {"screened", "validated", "rejected"},
            "screened": {"validated", "rejected"},
            "validated": {"champion", "rejected"},
            "champion": {"deployed"},
            "deployed": set(),
            "rejected": set(),
        }
        if state not in allowed[current.lifecycle_state]:
            raise ValueError(
                f"invalid lifecycle transition {current.lifecycle_state!r} -> {state!r}"
            )
        updated = replace(
            current,
            lifecycle_state=state,  # type: ignore[arg-type]
            artifact_uris=artifact_uris or current.artifact_uris,
            metrics=metrics or current.metrics,
        )
        rel = f"revisions/{updated.lifecycle_state}-{updated.sha}.json"
        run_dir = self.root / "runs" / run_id
        _write_new(run_dir / rel, updated.to_dict())
        _atomic_write(run_dir / "current.json", {"record": rel})
        return updated

    def record_artifacts(
        self, run_id: str, artifact_uris: tuple[str, ...]
    ) -> RunManifest:
        """Append an artifact revision without changing lifecycle state."""
        if not artifact_uris:
            raise ValueError("at least one artifact URI is required")
        current = self.load_run(run_id)
        updated = replace(current, artifact_uris=artifact_uris)
        rel = f"revisions/{updated.lifecycle_state}-artifacts-{updated.sha}.json"
        run_dir = self.root / "runs" / run_id
        _write_new(run_dir / rel, updated.to_dict())
        _atomic_write(run_dir / "current.json", {"record": rel})
        return updated

    def record_run_metadata(
        self,
        run_id: str,
        *,
        recipe: dict[str, Any] | None = None,
        hardware: dict[str, Any] | None = None,
        artifact_uris: tuple[str, ...] | None = None,
        legacy_kind: str | None = None,
        trace_id: str | None = None,
    ) -> RunManifest:
        """Append orchestration metadata without changing lifecycle state."""
        current = self.load_run(run_id)
        updated = replace(
            current,
            recipe=recipe or current.recipe,
            recipe_sha=content_sha(recipe) if recipe else current.recipe_sha,
            hardware=hardware or current.hardware,
            artifact_uris=artifact_uris or current.artifact_uris,
            legacy_kind=legacy_kind or current.legacy_kind,  # type: ignore[arg-type]
            trace_id=trace_id or current.trace_id,
        )
        rel = f"revisions/{updated.lifecycle_state}-metadata-{updated.sha}.json"
        run_dir = self.root / "runs" / run_id
        _write_new(run_dir / rel, updated.to_dict())
        _atomic_write(run_dir / "current.json", {"record": rel})
        return updated

    def write_snapshot(self, snapshot: DataSnapshot) -> Path:
        return _write_new(
            self.root
            / "data_snapshots"
            / f"{snapshot.snapshot_id}-{snapshot.sha}.json",
            snapshot.to_dict(),
        )

    def load_snapshot(self, sha_or_path: str | Path) -> DataSnapshot:
        path = Path(sha_or_path)
        if not path.exists():
            matches = list((self.root / "data_snapshots").glob(f"*-{sha_or_path}.json"))
            if len(matches) != 1:
                raise FileNotFoundError(f"data snapshot not found: {sha_or_path}")
            path = matches[0]
        data = _read(path)
        data["sources"] = tuple(data.get("sources") or ())
        return DataSnapshot(**data)

    def write_report(self, report: EvaluationReport) -> Path:
        return _write_new(
            self.root / "evaluations" / f"{report.report_id}-{report.sha}.json",
            report.to_dict(),
        )

    def load_report(self, sha_or_path: str | Path) -> EvaluationReport:
        path = Path(sha_or_path)
        if not path.exists():
            matches = list((self.root / "evaluations").glob(f"*-{sha_or_path}.json"))
            if len(matches) != 1:
                raise FileNotFoundError(f"evaluation report not found: {sha_or_path}")
            path = matches[0]
        return EvaluationReport(**_read(path))

    def write_merge(self, manifest: MergeManifest) -> Path:
        return _write_new(
            self.root / "merges" / f"{manifest.merge_id}-{manifest.sha}.json",
            manifest.to_dict(),
        )

    def promote(self, pointer: ChampionPointer) -> Path:
        track_dir = self.root / "champions" / pointer.track
        record = _write_new(
            track_dir / "history" / f"{pointer.pointer_id}-{pointer.sha}.json",
            pointer.to_dict(),
        )
        _atomic_write(
            track_dir / "current.json", {"record": str(record.relative_to(track_dir))}
        )
        return record

    def champion(self, track: str) -> ChampionPointer | None:
        track_dir = self.root / "champions" / track
        current = track_dir / "current.json"
        if not current.exists():
            return None
        data = _read(track_dir / _read(current)["record"])
        return ChampionPointer(**data)

    def deploy(self, pointer: ChampionPointer) -> Path:
        """Write a track deployment and the app-visible selected model atomically."""
        track_dir = self.root / "deployments" / pointer.track
        record = _write_new(
            track_dir / "history" / f"{pointer.pointer_id}-{pointer.sha}.json",
            pointer.to_dict(),
        )
        _atomic_write(
            track_dir / "current.json", {"record": str(record.relative_to(track_dir))}
        )
        _atomic_write(
            self.root / "deployments" / "selected.json",
            {"track": pointer.track, "record": str(record.relative_to(self.root))},
        )
        return record

    def lock_base(self, manifest: RunManifest) -> Path:
        """Permanently lock a track base; later calls may only confirm it."""
        track_dir = self.root / "tracks" / manifest.track / "base"
        current = track_dir / "current.json"
        payload = {
            "track": manifest.track,
            "run_id": manifest.run_id,
            "base_model_id": manifest.base_model_id,
            "base_model_revision": manifest.base_model_revision,
            "manifest_sha": manifest.sha,
            "created_at": utc_now(),
        }
        if current.exists():
            existing = _read(track_dir / _read(current)["record"])
            identity = (existing["base_model_id"], existing["base_model_revision"])
            requested = (manifest.base_model_id, manifest.base_model_revision)
            if identity != requested:
                raise ValueError(
                    f"{manifest.track} base is permanently locked to {identity}"
                )
            return track_dir / _read(current)["record"]
        sha = content_sha(payload)
        record = _write_new(track_dir / "history" / f"{sha}.json", payload)
        _atomic_write(
            track_dir / "current.json", {"record": str(record.relative_to(track_dir))}
        )
        return record
