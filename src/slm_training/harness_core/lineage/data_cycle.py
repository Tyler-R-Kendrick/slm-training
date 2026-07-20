"""Immutable corpus snapshot and feedback-cycle helpers."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from slm_training.harness_core.lineage.records import DataSnapshot, content_sha
from slm_training.harness_core.lineage.store import utc_now

if TYPE_CHECKING:
    from slm_training.harness_core.lineage.store import LineageStore


@dataclass(frozen=True)
class CycleData:
    sft_positives: tuple[dict[str, Any], ...]
    dpo_pairs: tuple[dict[str, Any], ...]
    verifier_negatives: tuple[dict[str, Any], ...]

    @property
    def sft_ready(self) -> bool:
        return len(self.sft_positives) >= 25

    @property
    def dpo_ready(self) -> bool:
        return len(self.dpo_pairs) >= 100


def sample_on_policy_replay(
    new_rows: Sequence[Mapping[str, Any]],
    validated_history: Sequence[Mapping[str, Any]],
    *,
    fraction: float = 0.10,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Mix champion history so replay is ``fraction`` of the final snapshot."""
    if not 0 <= fraction < 1:
        raise ValueError("replay fraction must be in [0, 1)")
    rows = [dict(row) for row in new_rows]
    if not rows or not validated_history or fraction == 0:
        return rows
    count = min(
        len(validated_history),
        max(1, math.ceil(len(rows) * fraction / (1 - fraction))),
    )
    history = sorted((dict(row) for row in validated_history), key=_row_identity)
    replay = random.Random(seed).sample(history, count)
    for row in replay:
        meta = dict(row.get("meta") or {})
        meta["on_policy_replay"] = True
        row["meta"] = meta
    return rows + replay


def _row_identity(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or content_sha(dict(row)))


def annotations_to_cycle_data(
    annotations: Iterable[Mapping[str, Any]],
    generation_attempts: Iterable[Mapping[str, Any]] = (),
) -> CycleData:
    """Convert feedback without promoting invalid output into SFT or DPO."""
    rows = [dict(row) for row in annotations]
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    valid_by_prompt: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        valid = row.get("valid") is not False
        corrected = bool(row.get("human_corrected"))
        approved = row.get("rating") == "up"
        if valid and (approved or corrected):
            positives.append(_training_row(row))
        if row.get("valid") is False:
            negatives.append(_negative_row(row))
        if valid and row.get("rating") in {"up", "down"}:
            prompt = str(row.get("prompt") or "")
            valid_by_prompt.setdefault(prompt, {"up": [], "down": []})[
                str(row["rating"])
            ].append(row)

    pairs: list[dict[str, Any]] = []
    for prompt, ratings in sorted(valid_by_prompt.items()):
        for chosen, rejected in zip(ratings["up"], ratings["down"], strict=False):
            pairs.append(
                {
                    "id": f"pair_{content_sha([_row_identity(chosen), _row_identity(rejected)])[:16]}",
                    "prompt": prompt,
                    "chosen": str(chosen.get("openui") or ""),
                    "rejected": str(rejected.get("openui") or ""),
                    "chosen_id": _row_identity(chosen),
                    "rejected_id": _row_identity(rejected),
                    "identities": {
                        "chosen": dict(chosen.get("identities") or {}),
                        "rejected": dict(rejected.get("identities") or {}),
                    },
                }
            )
    for attempt in generation_attempts:
        if not bool(attempt.get("valid")):
            negatives.append(_negative_row(attempt))
    return CycleData(tuple(positives), tuple(pairs), tuple(negatives))


def _training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": _row_identity(row),
        "prompt": str(row.get("prompt") or ""),
        "openui": str(row.get("openui") or ""),
        "design_md": row.get("design_md"),
        "parent": row.get("checkpoint") or (row.get("meta") or {}).get("parent"),
        "generator": (row.get("identities") or {}).get("output_generator"),
        "reviewer": (row.get("identities") or {}).get("reviewer"),
        "annotator": (row.get("identities") or {}).get("annotation_author"),
        "correction_author": (row.get("identities") or {}).get("correction_author"),
        "generation_id": row.get("generation_id"),
        "human_corrected": bool(row.get("human_corrected")),
    }


def _negative_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": _row_identity(row),
        "prompt": str(row.get("prompt") or ""),
        "openui": str(row.get("openui") or ""),
        "error": row.get("error"),
        "checkpoint": row.get("checkpoint"),
        "identities": dict(row.get("identities") or {}),
    }


def snapshot_directory(
    snapshot_id: str,
    sources: Sequence[Path | str],
    *,
    target_token_count: int | None = None,
    annotations_sha: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DataSnapshot:
    """Hash source bytes in stable path order; source files remain untouched."""
    files: list[Path] = []
    for source in map(Path, sources):
        if source.is_dir():
            files.extend(path for path in source.rglob("*") if path.is_file())
        elif source.is_file():
            files.append(source)
        else:
            raise FileNotFoundError(source)
    files = sorted(set(files), key=lambda path: str(path))
    inventory = [
        {
            "path": str(path),
            "sha": content_sha(path.read_bytes().hex()),
            "size": path.stat().st_size,
        }
        for path in files
    ]
    record_count = sum(_jsonl_rows(path) for path in files if path.suffix == ".jsonl")
    if target_token_count is None:
        target_token_count = sum(
            _jsonl_target_tokens(path) for path in files if path.suffix == ".jsonl"
        )
    return DataSnapshot(
        snapshot_id=snapshot_id,
        sources=tuple(str(source) for source in sources),
        records_sha=content_sha(inventory),
        record_count=record_count,
        target_token_count=int(target_token_count),
        annotations_sha=annotations_sha,
        created_at=utc_now(),
        metadata={**dict(metadata or {}), "files": inventory},
    )


def register_dataset_snapshot(
    store: "LineageStore",
    *,
    dataset_dir: Path,
    kind: str,
    snapshot_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[DataSnapshot, Path, bool]:
    """Register a built dataset directory as a lineage DataSnapshot.

    Keyed on the dataset's stable content fingerprint (manifest
    ``content_fingerprint``; records hash fallback), NOT on file bytes, so
    re-running an identical build reuses the existing snapshot instead of
    piling up timestamp-only variants. Returns (snapshot, path, created).
    """
    from slm_training.data.store import dataset_fingerprint

    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = dataset_fingerprint(dataset_dir)
    if not fingerprint:
        raise ValueError(f"dataset has no content fingerprint: {dataset_dir}")
    snapshot_id = snapshot_id or f"{kind}-{manifest.get('version') or dataset_dir.name}"

    snapshot_root = store.root / "data_snapshots"
    if snapshot_root.is_dir():
        for existing in sorted(snapshot_root.glob(f"{snapshot_id}-*.json")):
            payload = json.loads(existing.read_text(encoding="utf-8"))
            if payload.get("records_sha") == fingerprint:
                payload["sources"] = tuple(payload.get("sources") or ())
                return DataSnapshot(**payload), existing, False

    record_count = int(manifest.get("record_count") or 0)
    if not record_count:
        record_count = sum(
            _jsonl_rows(path) for path in sorted(dataset_dir.rglob("records.jsonl"))
        )
    target_token_count = sum(
        _jsonl_target_tokens(path)
        for path in sorted(dataset_dir.rglob("records.jsonl"))
    )
    snapshot = DataSnapshot(
        snapshot_id=snapshot_id,
        sources=(str(dataset_dir),),
        records_sha=fingerprint,
        record_count=record_count,
        target_token_count=int(target_token_count),
        created_at=utc_now(),
        metadata={
            "kind": kind,
            "manifest": str(manifest_path) if manifest_path.is_file() else None,
            "profile": manifest.get("profile"),
            "trace_id": manifest.get("trace_id"),
            **dict(metadata or {}),
        },
    )
    path = store.write_snapshot(snapshot)
    return snapshot, path, True


def _jsonl_rows(path: Path) -> int:
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _jsonl_target_tokens(path: Path) -> int:
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            target = str(json.loads(line).get("openui") or "")
        except json.JSONDecodeError:
            continue
        total += len(target.encode("utf-8"))
    return total
