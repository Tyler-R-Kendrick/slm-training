"""Canonical local-first model-data storage and explicit Git publication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DataKind = Literal[
    "train",
    "eval",
    "preference",
    "annotation",
    "trajectory",
    "programspec",
    "mixture",
    "solver_supervision",
]
DATA_KINDS: tuple[DataKind, ...] = (
    "train",
    "eval",
    "preference",
    "annotation",
    "trajectory",
    "programspec",
    "mixture",
    "solver_supervision",
)
PUBLISHABLE_KINDS = frozenset(DATA_KINDS) - {"annotation", "solver_supervision"}
MAX_GIT_FILE_BYTES = 50 * 1024 * 1024
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_LEGACY_ROOTS: dict[DataKind, Path] = {
    "train": Path("outputs/train_data"),
    "eval": Path("outputs/test_data"),
    "preference": Path("outputs/preferences"),
    "annotation": Path("outputs/annotations"),
    "trajectory": Path("outputs/traces"),
    "programspec": Path("outputs/progspec"),
    "mixture": Path("outputs/mixtures"),
    "solver_supervision": Path("outputs/solver_supervision"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(path: Path) -> dict[str, Any]:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return {}
    return json.loads(manifest.read_text(encoding="utf-8"))


def dataset_fingerprint(path: Path) -> str | None:
    payload = _manifest(path)
    for key in ("content_fingerprint", "records_sha", "manifest_sha256"):
        if payload.get(key):
            return str(payload[key])
    records = path / "records.jsonl"
    return _sha256(records) if records.is_file() else None


@dataclass(frozen=True)
class DatasetRef:
    kind: DataKind
    dataset_id: str
    path: Path
    storage: Literal["local", "git", "legacy"]
    fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset_id": self.dataset_id,
            "path": self.path.as_posix(),
            "storage": self.storage,
            "fingerprint": self.fingerprint,
        }


class DataStore:
    """Resolve model data locally first, then from immutable Git resources."""

    def __init__(
        self,
        root: Path | str = Path("."),
        *,
        local_root: Path | str | None = None,
        published_root: Path | str | None = None,
    ) -> None:
        self.root = Path(root)
        configured = local_root or os.getenv("SLM_DATA_ROOT") or Path("outputs/data")
        self.local_root = self._under_root(configured)
        self.published_root = self._under_root(
            published_root or Path("src/slm_training/resources/data")
        )

    def _under_root(self, value: Path | str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def validate_id(dataset_id: str) -> str:
        if not _ID_RE.fullmatch(dataset_id):
            raise ValueError(f"invalid dataset id: {dataset_id!r}")
        return dataset_id

    def path(self, kind: DataKind, dataset_id: str) -> Path:
        self._validate(kind, dataset_id)
        return self.local_root / kind / dataset_id

    def published_path(self, kind: DataKind, dataset_id: str) -> Path:
        self._validate(kind, dataset_id)
        return self.published_root / kind / dataset_id

    def legacy_path(self, kind: DataKind, dataset_id: str) -> Path:
        self._validate(kind, dataset_id)
        return self.root / _LEGACY_ROOTS[kind] / dataset_id

    def resolve(self, kind: DataKind, dataset_id: str) -> DatasetRef:
        self._validate(kind, dataset_id)
        local = self.path(kind, dataset_id)
        published = self.published_path(kind, dataset_id)
        if local.exists() and published.exists():
            local_fp = dataset_fingerprint(local)
            published_fp = dataset_fingerprint(published)
            if local_fp != published_fp:
                raise ValueError(
                    f"dataset {kind}:{dataset_id} differs between local and Git stores"
                )
        for storage, path in (
            ("local", local),
            ("git", published),
            ("legacy", self.legacy_path(kind, dataset_id)),
        ):
            if path.exists():
                return DatasetRef(
                    kind, dataset_id, path, storage, dataset_fingerprint(path)  # type: ignore[arg-type]
                )
        raise FileNotFoundError(f"dataset not found: {kind}:{dataset_id}")

    def resolve_path(self, kind: DataKind, value: Path | str) -> Path:
        """Keep explicit existing paths; resolve missing conventional paths by ID."""
        path = Path(value)
        if path.exists():
            return path
        dataset_id = path.name if len(path.parts) > 1 else str(path)
        try:
            return self.resolve(kind, dataset_id).path
        except FileNotFoundError:
            return path

    def versions(self, kind: DataKind) -> list[DatasetRef]:
        found: dict[str, DatasetRef] = {}
        roots = (
            ("legacy", self.root / _LEGACY_ROOTS[kind]),
            ("git", self.published_root / kind),
            ("local", self.local_root / kind),
        )
        for storage, base in roots:
            if not base.exists():
                continue
            for path in sorted(item for item in base.iterdir() if item.is_dir()):
                if kind == "trajectory" and storage == "legacy" and path.name != "latest":
                    continue
                if _ID_RE.fullmatch(path.name):
                    found[path.name] = DatasetRef(
                        kind,
                        path.name,
                        path,
                        storage,  # type: ignore[arg-type]
                        dataset_fingerprint(path),
                    )
        return [found[key] for key in sorted(found)]

    def verify(self, kind: DataKind, dataset_id: str) -> DatasetRef:
        ref = self.resolve(kind, dataset_id)
        manifest = _manifest(ref.path)
        if not manifest:
            raise ValueError(f"dataset lacks manifest.json: {ref.path}")
        declared = str(manifest.get("kind") or "")
        expected = {kind, f"{kind}_data", "test_data" if kind == "eval" else kind}
        if declared and declared not in expected:
            raise ValueError(f"dataset kind mismatch: expected {kind}, found {declared}")
        for artifact in manifest.get("artifacts") or []:
            relative = Path(str(artifact["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe artifact path: {relative}")
            path = ref.path / relative
            if not path.is_file() or _sha256(path) != artifact.get("sha256"):
                raise ValueError(f"artifact hash mismatch: {relative}")
        return ref

    def publish(self, kind: DataKind, dataset_id: str) -> DatasetRef:
        if kind not in PUBLISHABLE_KINDS:
            raise ValueError(f"{kind} data must be snapshotted before publication")
        source = self.resolve(kind, dataset_id)
        if source.storage == "git":
            return source
        self.verify(kind, dataset_id)
        destination = self.published_path(kind, dataset_id)
        if destination.exists():
            raise FileExistsError(f"published dataset is immutable: {destination}")
        files = [path for path in source.path.rglob("*") if path.is_file()]
        for path in files:
            if path.is_symlink():
                raise ValueError(f"dataset cannot publish symlink: {path}")
            if path.stat().st_size >= MAX_GIT_FILE_BYTES:
                raise ValueError(f"dataset file exceeds Git cap: {path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source.path, destination)
        source_manifest = _manifest(destination)
        write_common_manifest(
            destination,
            kind=kind,
            dataset_id=dataset_id,
            trace_id=source_manifest.get("trace_id"),
            immutable=True,
        )
        return DatasetRef(
            kind, dataset_id, destination, "git", dataset_fingerprint(destination)
        )

    def migration_plan(self) -> list[tuple[Path, Path]]:
        plan: list[tuple[Path, Path]] = []
        for kind, legacy in _LEGACY_ROOTS.items():
            source = self.root / legacy
            if kind == "trajectory":
                source = source / "latest"
            if not source.exists():
                continue
            destination = self.local_root / kind
            if kind == "trajectory":
                destination = destination / "latest"
            if not destination.exists():
                plan.append((source, destination))
        return plan

    def migrate(self) -> list[tuple[Path, Path]]:
        plan = self.migration_plan()
        for source, destination in plan:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
        return plan

    @staticmethod
    def _validate(kind: DataKind, dataset_id: str) -> None:
        if kind not in DATA_KINDS:
            raise ValueError(f"unknown dataset kind: {kind}")
        DataStore.validate_id(dataset_id)


def write_common_manifest(
    directory: Path,
    *,
    kind: DataKind,
    dataset_id: str,
    trace_id: str | None = None,
    immutable: bool = False,
) -> dict[str, Any]:
    """Add the common storage envelope without discarding domain metadata."""
    manifest_path = directory / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path == manifest_path or path.is_symlink():
            continue
        artifacts.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
        )
    payload.update(
        {
            "schema_version": 2,
            "dataset_id": dataset_id,
            "kind": kind,
            "immutable": immutable,
            "trace_id": trace_id,
            "artifacts": artifacts,
        }
    )
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DATA_KINDS",
    "MAX_GIT_FILE_BYTES",
    "DataKind",
    "DataStore",
    "DatasetRef",
    "dataset_fingerprint",
    "write_common_manifest",
]
