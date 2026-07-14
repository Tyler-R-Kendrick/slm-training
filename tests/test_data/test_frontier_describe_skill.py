"""Contract tests for the frontier-describe agent skill (P5 / SLM-9)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from slm_training.data.frontier import gold_content_hash


def _module():
    path = Path(".agents/skills/frontier-describe/scripts/finalize.py")
    spec = importlib.util.spec_from_file_location("frontier_describe_finalize", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row() -> dict:
    return {
        "gold_id": "train_a",
        "gold_content_hash": gold_content_hash(
            'root = TextContent(":copy.line")', "Create one text element."
        ),
        "prompt": "Create one text element.",
        "skeleton_openui": 'root = TextContent(":copy.line")',
        "has_fresh_artifact": False,
    }


def _bundle(row: dict) -> dict:
    module = _module()
    return {
        "gold_id": row["gold_id"],
        "gold_content_hash": row["gold_content_hash"],
        "skeleton_openui": row["skeleton_openui"],
        "provenance": {
            "skill": module.SKILL,
            "skill_version": module.SKILL_VERSION,
            "prompt_hash": module._prompt_hash(row["prompt"]),
            "generated_at": "2026-07-14T00:00:00Z",
        },
        "paraphrases": ["Show :copy.line as a single text element."],
        "ladder": {
            f"L{i}": f"Level {i} description for :copy.line." for i in range(1, 6)
        },
        "edits": [
            {
                "edit_op": "remove",
                "instruction": "Remove :copy.line.",
                "delta_ref": "root:remove",
            }
        ],
        "vision": {},
    }


def test_validator_accepts_placeholder_safe_bundle() -> None:
    module = _module()
    row = _row()
    assert module.validate_bundle(_bundle(row), row) == []


def test_validator_rejects_dsl_literal_copy_and_unknown_placeholder() -> None:
    module = _module()
    row = _row()
    bundle = _bundle(row)
    bundle["paraphrases"] = [
        row["prompt"],
        'root = TextContent(":copy.line")',
        "Show :unknown.slot.",
    ]
    errors = module.validate_bundle(bundle, row)
    assert any("copies the source prompt" in error for error in errors)
    assert any("contains OpenUI DSL" in error for error in errors)
    assert any("introduces placeholders" in error for error in errors)


def test_manifest_is_deterministic_and_stale_safe(tmp_path: Path) -> None:
    module = _module()
    row = _row()
    worklist = tmp_path / "worklist.jsonl"
    worklist.write_text(json.dumps(row) + "\n", encoding="utf-8")

    pending, errors = module.build_manifest(worklist, tmp_path)
    assert errors == []
    assert pending["pending"] == 1

    path = tmp_path / f"{row['gold_id']}.{row['gold_content_hash']}.json"
    path.write_text(json.dumps(_bundle(row)), encoding="utf-8")
    complete, errors = module.build_manifest(worklist, tmp_path)
    assert errors == []
    assert complete["complete"] == 1
    assert complete == module.build_manifest(worklist, tmp_path)[0]

    row["prompt"] = "Changed prompt."
    worklist.write_text(json.dumps(row) + "\n", encoding="utf-8")
    stale, errors = module.build_manifest(worklist, tmp_path)
    assert stale["invalid"] == 1
    assert any("stale" in error for error in errors)
