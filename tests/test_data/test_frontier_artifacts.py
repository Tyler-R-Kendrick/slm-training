"""Frozen frontier artifact contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.frontier import (
    artifact_path,
    gold_content_hash,
    prompt_hash,
    write_worklist,
)
from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.synth import FrozenArtifactSynthesizer

OPENUI = 'root = Stack([cta], "column")\ncta = Button(":cta.label")'


def _gold(*, prompt: str = "Button", split: str = "train") -> ExampleRecord:
    return ExampleRecord(
        id="gold_1",
        prompt=prompt,
        openui=OPENUI,
        placeholders=[":cta.label"],
        split=split,
        meta={
            "program_family_id": "family_1",
            "lineage_id": "lineage_1",
            "split_group_id": "group_1",
        },
    )


def _bundle(gold: ExampleRecord, **updates: object) -> dict:
    data = {
        "schema_version": 1,
        "gold_id": gold.id,
        "gold_content_hash": gold_content_hash(gold.openui, gold.prompt),
        "skeleton_openui": gold.openui,
        "provenance": {
            "skill_name": "frontier-describe",
            "skill_version": "1",
            "prompt_hash": prompt_hash(gold.prompt),
            "generated_at": "2026-07-14T00:00:00Z",
        },
        "paraphrases": ["Create a CTA button"],
        "ladder": [{"prompt": "A call to action", "level": "L4"}],
        "edits": [{"prompt": "Keep the CTA", "openui": gold.openui}],
        "vision": [{"prompt": "Match the CTA screenshot"}],
    }
    data.update(updates)
    return data


def _write_bundle(root: Path, gold: ExampleRecord, **updates: object) -> Path:
    path = artifact_path(root, gold)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_bundle(gold, **updates)), encoding="utf-8")
    return path


def test_gold_hash_ignores_style_but_binds_prompt() -> None:
    styled = 'root = Stack([cta], "column", "m")\ncta = Button(":cta.label")'
    assert gold_content_hash(styled, "Button") == gold_content_hash(OPENUI, "Button")
    assert gold_content_hash(OPENUI, "Button") != gold_content_hash(OPENUI, "Other")


def test_worklist_is_train_only_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "frontier"
    train = _gold()
    held = _gold(split="held_out")
    first = write_worklist([held, train], root=root)
    before = (root / "worklist.jsonl").read_bytes()
    second = write_worklist([held, train], root=root)
    assert first == second
    assert (root / "worklist.jsonl").read_bytes() == before
    assert first["gold_count"] == 1
    assert first["pending_count"] == 1

    _write_bundle(root, train)
    completed = write_worklist([held, train], root=root)
    assert completed["complete_count"] == 1
    assert completed["pending_count"] == 0
    assert (root / "worklist.jsonl").read_text(encoding="utf-8") == ""


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge deps missing")
def test_reader_emits_all_blocks_and_drops_stale(tmp_path: Path) -> None:
    gold = _gold()
    _write_bundle(tmp_path, gold)
    synth = FrozenArtifactSynthesizer(tmp_path)
    records = synth.expand(gold)
    assert len(records) == 4
    assert {r.meta["frontier"]["kind"] for r in records} == {
        "paraphrases",
        "ladder",
        "edits",
        "vision",
    }
    assert all(r.meta["split_group_id"] == "group_1" for r in records)
    assert all(r.meta["tier"] == "Bronze" for r in records)
    assert all(r.meta["verification_tier"] == "Bronze" for r in records)
    assert all(r.meta["failing_gate"] is None for r in records)

    _write_bundle(
        tmp_path,
        gold,
        edits=[{"prompt": "Invalid edit", "openui": "root = Broken()"}],
    )
    revalidated = synth.expand(gold)
    assert len(revalidated) == 3
    assert all(r.meta["frontier"]["kind"] != "edits" for r in revalidated)
    assert synth.expand(_gold(prompt="Changed prompt")) == []
    assert synth.expand(_gold(split="held_out")) == []


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge deps missing")
def test_reader_rejects_structurally_unfaithful_bundle(tmp_path: Path) -> None:
    gold = _gold()
    _write_bundle(
        tmp_path,
        gold,
        skeleton_openui='root = TextContent(":different.text")',
    )
    assert FrozenArtifactSynthesizer(tmp_path).expand(gold) == []
