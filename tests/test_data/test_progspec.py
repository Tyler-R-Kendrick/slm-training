"""ProgramSpec round-trip, projection, and split-group leakage tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from slm_training.data.leakage import find_leakage, load_train_fingerprints
from slm_training.data.progspec import ProgramSpec, emit_record
from slm_training.dsl import bridge_available
from slm_training.dsl.language_contract import contract_id
from slm_training.dsl.schema import TASK_TOKENS, ExampleRecord, write_jsonl
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
from slm_training.harnesses.train_data.catalog import classify_source_family

OPENUI = 'root = Stack([cta])\ncta = Button(":cta.label")'


def test_language_contract_can_load_before_generator_exports() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import slm_training.data.language_contract; "
                "from slm_training.data.progspec import ProgramGenerator; "
                "assert ProgramGenerator.__name__ == 'ProgramGenerator'"
            ),
        ],
        check=True,
    )


def _spec() -> ProgramSpec:
    return ProgramSpec(
        id="program_1",
        ast={"type": "root"},
        canonical_openui=OPENUI,
        facts={"components": ["Stack", "Button"]},
        contract_id=contract_id(),
        program_family_id="family_1",
        lineage_id="lineage_1",
        split_group_id="group_1",
        split="train",
        derivative_refs=("render_1",),
        provenance={"source": "unit"},
    )


def test_programspec_round_trips() -> None:
    spec = _spec()
    assert ProgramSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge deps missing")
def test_programspec_from_openui_uses_current_contract() -> None:
    spec = ProgramSpec.from_openui(
        id="program_2",
        openui=OPENUI,
        facts={},
        program_family_id="family_2",
        lineage_id="lineage_2",
        split_group_id="group_2",
    )
    assert spec.ast
    assert spec.contract_id == contract_id()


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge deps missing")
def test_emit_record_supports_every_task_and_inherits_split() -> None:
    spec = _spec()
    for task in TASK_TOKENS:
        record = emit_record(spec, prompt=f"do {task}", task=task)
        assert record.split == "train"
        assert record.meta["split_group_id"] == "group_1"
        assert record.meta["contract_id"] == spec.contract_id
        assert record.meta["task"] == task
        assert record.meta["parent_id"] == spec.id
        assert classify_source_family(record) == "programspec_generated"


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge deps missing")
def test_emit_record_runs_verifier_and_rejects_stale_contract() -> None:
    seen: list[str] = []
    record = emit_record(
        _spec(),
        prompt="make a button",
        task="generation",
        verifier=lambda value: seen.append(value.id),
    )
    assert seen == [record.id]
    stale = ProgramSpec.from_dict({**_spec().to_dict(), "contract_id": "0" * 16})
    with pytest.raises(ValueError, match="stale ProgramSpec contract"):
        emit_record(stale, prompt="p", task="generation")


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge deps missing")
def test_legacy_train_build_does_not_invent_lineage(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.jsonl"
    write_jsonl(
        seeds,
        [
            ExampleRecord(
                id="legacy",
                prompt="Button",
                openui=OPENUI,
                split="train",
            )
        ],
    )
    result = build_train_data(
        TrainDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "out",
            synthesizer="none",
        )
    )
    row = json.loads(
        (Path(result["output_dir"]) / "records.jsonl").read_text(encoding="utf-8")
    )
    assert "task" not in row["meta"]
    assert "split_group_id" not in row["meta"]
    assert result["manifest"]["split_group_ids"] == []


def test_split_group_overlap_is_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "ids": ["train_1"],
                "split_group_ids": ["group_1"],
                "prompt_fingerprints": ["present"],
                "openui_fingerprints": ["present"],
                "structure_fingerprints": ["present"],
                "pair_fingerprints": ["present"],
                "design_md_fingerprints": ["present"],
            }
        ),
        encoding="utf-8",
    )
    held = ExampleRecord(
        id="held_1",
        prompt="different",
        openui='root = TextContent(":held.text")',
        split="held_out",
        meta={"split_group_id": "group_1"},
    )
    reasons = find_leakage(held, load_train_fingerprints(manifest))
    assert "split_group_id" in reasons


def test_meta_subschemas_are_dicts() -> None:
    with pytest.raises(ValueError, match="meta.repair"):
        ExampleRecord(
            id="bad",
            prompt="p",
            openui=OPENUI,
            meta={"repair": "not-a-record"},
        )
