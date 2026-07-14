"""Deterministic worklist and reader for committed frontier bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from slm_training.data.frontier.hashing import gold_content_hash, prompt_hash
from slm_training.data.leakage import fingerprint_openui_structure
from slm_training.data.structure import strip_style_literals
from slm_training.dsl.schema import ExampleRecord

SCHEMA_VERSION = 1
_BLOCKS = ("paraphrases", "ladder", "edits", "vision")


def artifact_path(root: Path | str, record: ExampleRecord) -> Path:
    digest = gold_content_hash(record.openui, record.prompt)
    return Path(root) / f"{record.id}.{digest[:8]}.json"


def load_bundle(path: Path | str, gold: ExampleRecord) -> dict[str, Any] | None:
    """Return a faithful current bundle, or ``None`` for stale/invalid input."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        expected_hash = gold_content_hash(gold.openui, gold.prompt)
        if data.get("schema_version") != SCHEMA_VERSION:
            return None
        if data.get("gold_id") != gold.id or data.get("gold_content_hash") != expected_hash:
            return None
        if fingerprint_openui_structure(str(data["skeleton_openui"])) != (
            fingerprint_openui_structure(gold.openui)
        ):
            return None
        provenance = data.get("provenance")
        if not isinstance(provenance, dict):
            return None
        if provenance.get("prompt_hash") != prompt_hash(gold.prompt):
            return None
        if not all(provenance.get(key) for key in ("skill_name", "skill_version", "generated_at")):
            return None
        if not all(isinstance(data.get(block), list) for block in _BLOCKS):
            return None
        return data
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def build_worklist(
    records: Iterable[ExampleRecord],
    *,
    root: Path | str,
    skill_name: str = "frontier-describe",
    skill_version: str = "1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """List train golds whose current hash-bound artifact is absent or invalid."""
    root = Path(root)
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, str]] = {}
    train = sorted((record for record in records if record.split == "train"), key=lambda r: r.id)
    for record in train:
        digest = gold_content_hash(record.openui, record.prompt)
        path = artifact_path(root, record)
        complete = load_bundle(path, record) is not None
        artifacts[record.id] = {
            "gold_content_hash": digest,
            "path": path.name,
            "status": "complete" if complete else "pending",
        }
        if complete:
            continue
        meta = record.meta or {}
        rows.append(
            {
                "gold_id": record.id,
                "gold_content_hash": digest,
                "prompt_hash": prompt_hash(record.prompt),
                "prompt": record.prompt,
                "placeholder_skeleton": strip_style_literals(record.openui or "").strip(),
                "placeholders": list(record.placeholders),
                "program_family_id": str(meta.get("program_family_id") or record.id),
                "lineage_id": str(meta.get("lineage_id") or record.id),
                "split_group_id": str(meta.get("split_group_id") or record.id),
                "provenance": {
                    "skill_name": skill_name,
                    "skill_version": skill_version,
                    "generated_at": None,
                },
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "gold_count": len(train),
        "complete_count": sum(v["status"] == "complete" for v in artifacts.values()),
        "pending_count": len(rows),
        "artifacts": artifacts,
    }
    return rows, manifest


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def write_worklist(
    records: Iterable[ExampleRecord],
    *,
    root: Path | str,
    worklist_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    skill_name: str = "frontier-describe",
    skill_version: str = "1",
) -> dict[str, Any]:
    root = Path(root)
    rows, manifest = build_worklist(
        records,
        root=root,
        skill_name=skill_name,
        skill_version=skill_version,
    )
    worklist = Path(worklist_path) if worklist_path else root / "worklist.jsonl"
    manifest_file = Path(manifest_path) if manifest_path else root / "MANIFEST.json"
    worklist_text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    _write_if_changed(worklist, worklist_text)
    _write_if_changed(
        manifest_file,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return manifest
