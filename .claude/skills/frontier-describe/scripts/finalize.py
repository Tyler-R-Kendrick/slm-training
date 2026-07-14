#!/usr/bin/env python3
"""Validate frozen frontier bundles and write a deterministic coverage manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from slm_training.data.frontier import artifact_path, gold_content_hash

SKILL = "frontier-describe"
SKILL_VERSION = "0.1.0"
_PLACEHOLDER_RE = re.compile(r":[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")
_DSL_RE = re.compile(r"(?m)^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*[A-Z][A-Za-z0-9_]*\s*\(")
_QUOTED_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
_STRUCTURAL_LITERALS = {"column", "row", "horizontal", "vertical"}


def _norm(value: str) -> str:
    return " ".join(value.split())


def _prompt_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _generated_text(bundle: dict[str, Any]) -> list[str]:
    texts = [str(value) for value in bundle.get("paraphrases") or []]
    texts.extend(str(value) for value in (bundle.get("ladder") or {}).values())
    texts.extend(
        str(edit.get("instruction", ""))
        for edit in bundle.get("edits") or []
        if isinstance(edit, dict)
    )
    vision = bundle.get("vision") or {}
    if isinstance(vision, dict) and vision.get("semantic_description"):
        texts.append(str(vision["semantic_description"]))
    return texts


def validate_bundle(bundle: Any, row: dict[str, Any]) -> list[str]:
    """Return deterministic validation errors for one current worklist row."""
    if not isinstance(bundle, dict):
        return ["bundle must be a JSON object"]

    errors: list[str] = []
    expected = {
        "gold_id": row["gold_id"],
        "gold_content_hash": row["gold_content_hash"],
        "skeleton_openui": row["skeleton_openui"],
    }
    for key, value in expected.items():
        if bundle.get(key) != value:
            errors.append(f"{key} must match the worklist exactly")

    provenance = bundle.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    else:
        required = {
            "skill": SKILL,
            "skill_version": SKILL_VERSION,
            "prompt_hash": _prompt_hash(str(row.get("prompt", ""))),
        }
        for key, value in required.items():
            if provenance.get(key) != value:
                errors.append(f"provenance.{key} must equal {value!r}")
        if not provenance.get("generated_at"):
            errors.append("provenance.generated_at is required")

    paraphrases = bundle.get("paraphrases")
    if (
        not isinstance(paraphrases, list)
        or not paraphrases
        or not all(isinstance(value, str) and value.strip() for value in paraphrases)
    ):
        errors.append("paraphrases must be a non-empty string list")

    ladder = bundle.get("ladder")
    if not isinstance(ladder, dict) or list(ladder) != [f"L{i}" for i in range(1, 6)]:
        errors.append("ladder must contain ordered keys L1 through L5")
    elif not all(isinstance(value, str) and value.strip() for value in ladder.values()):
        errors.append("ladder values must be non-empty strings")

    edits = bundle.get("edits")
    if not isinstance(edits, list) or not edits:
        errors.append("edits must be a non-empty list")
    elif not all(
        isinstance(edit, dict)
        and edit.get("edit_op")
        and edit.get("instruction")
        and edit.get("delta_ref")
        for edit in edits
    ):
        errors.append("each edit needs edit_op, instruction, and delta_ref")

    vision = bundle.get("vision")
    if not isinstance(vision, dict):
        errors.append("vision must be an object")

    skeleton = str(row.get("skeleton_openui", ""))
    allowed_placeholders = set(_PLACEHOLDER_RE.findall(skeleton))
    copied_literals = {
        value
        for value in _QUOTED_RE.findall(skeleton)
        if not value.startswith(":") and value.lower() not in _STRUCTURAL_LITERALS
    }
    source_prompt = _norm(str(row.get("prompt", "")))
    for index, text in enumerate(_generated_text(bundle)):
        if _DSL_RE.search(text) or (skeleton.strip() and skeleton.strip() in text):
            errors.append(f"generated text {index} contains OpenUI DSL")
        if _norm(text) == source_prompt:
            errors.append(f"generated text {index} copies the source prompt")
        unknown = set(_PLACEHOLDER_RE.findall(text)) - allowed_placeholders
        if unknown:
            errors.append(
                f"generated text {index} introduces placeholders: {sorted(unknown)}"
            )
        for literal in copied_literals:
            if literal and literal in text:
                errors.append(f"generated text {index} copies literal {literal!r}")
    return errors


def load_worklist(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("worklist rows must be JSON objects")
    return sorted(rows, key=lambda row: str(row["gold_id"]))


def build_manifest(worklist: Path, root: Path) -> tuple[dict[str, Any], list[str]]:
    rows = load_worklist(worklist)
    entries: list[dict[str, Any]] = []
    all_errors: list[str] = []
    for row in rows:
        computed_hash = gold_content_hash(
            str(row.get("skeleton_openui", "")), str(row.get("prompt", ""))
        )
        if computed_hash != row.get("gold_content_hash"):
            errors = ["worklist gold_content_hash is stale"]
            path = artifact_path(
                str(row["gold_id"]), str(row["gold_content_hash"]), root=root
            )
        else:
            path = artifact_path(str(row["gold_id"]), computed_hash, root=root)
            if path.exists():
                try:
                    bundle = json.loads(path.read_text(encoding="utf-8"))
                    errors = validate_bundle(bundle, row)
                except (OSError, json.JSONDecodeError) as exc:
                    errors = [f"unreadable bundle: {exc}"]
            else:
                errors = []
        status = "invalid" if errors else ("complete" if path.exists() else "pending")
        entries.append(
            {
                "gold_id": row["gold_id"],
                "gold_content_hash": row["gold_content_hash"],
                "path": path.as_posix(),
                "status": status,
            }
        )
        all_errors.extend(f"{row['gold_id']}: {error}" for error in errors)

    counts = {
        status: sum(entry["status"] == status for entry in entries)
        for status in ("complete", "pending", "invalid")
    }
    manifest = {
        "schema_version": 1,
        "skill": SKILL,
        "skill_version": SKILL_VERSION,
        "worklist": worklist.as_posix(),
        "total": len(entries),
        **counts,
        "leakage": {
            "dsl": 0 if not any("DSL" in error for error in all_errors) else 1,
            "literal_copy": 0
            if not any("copies" in error for error in all_errors)
            else 1,
        },
        "artifacts": entries,
    }
    return manifest, all_errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worklist", type=Path, default=Path("fixtures/frontier/worklist.jsonl")
    )
    parser.add_argument("--root", type=Path, default=Path("fixtures/frontier"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("fixtures/frontier/MANIFEST.json")
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)

    manifest, errors = build_manifest(args.worklist, args.root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("total", "complete", "pending", "invalid", "leakage")
            },
            sort_keys=True,
        )
    )
    for error in errors:
        print(error)
    return int(bool(errors) or (args.require_complete and manifest["pending"] > 0))


if __name__ == "__main__":
    raise SystemExit(main())
