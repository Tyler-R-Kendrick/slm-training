"""Training-data quality + determinism tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.data.quality import assess_record
from slm_training.dsl import bridge_available, validate
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing",
)


def test_train_seeds_all_validate() -> None:
    records = load_jsonl("fixtures/train_seeds.jsonl")
    assert len(records) >= 18
    for record in records:
        program = validate(record.openui)
        assert program.placeholders
        assert record.openui.startswith("root =")


def test_tabs_content_array_not_flagged_as_placeholder_prop() -> None:
    src = (
        'root = Stack([tabs], "column", "m")\n'
        'body1 = TextContent(":tab.one.body")\n'
        'body2 = TextContent(":tab.two.body")\n'
        'i1 = TabItem("one", ":tab.one.trigger", [body1])\n'
        'i2 = TabItem("two", ":tab.two.trigger", [body2])\n'
        "tabs = Tabs([i1, i2])"
    )
    validate(src)


def test_quality_fixture_build_is_deterministic(tmp_path: Path) -> None:
    """Fixture-only builds are fast and bit-stable across runs."""
    cfg_kwargs = dict(
        seed_path=Path("fixtures/train_seeds.jsonl"),
        rico_path=None,
        source="fixture",
        synthesizer="quality",
        require_design_md=True,
    )
    first = build_train_data(
        TrainDataConfig(output_root=tmp_path / "a", version="v1", **cfg_kwargs)
    )
    second = build_train_data(
        TrainDataConfig(output_root=tmp_path / "b", version="v1", **cfg_kwargs)
    )
    assert first["manifest"]["content_fingerprint"] == second["manifest"]["content_fingerprint"]
    assert first["stats"]["record_count"] == second["stats"]["record_count"]
    assert first["stats"]["record_count"] >= 20
    assert first["stats"]["error_count"] == 0
    assert first["stats"]["mean_quality_score"] is not None
    assert first["stats"]["mean_quality_score"] >= 0.7
    assert first["stats"]["with_design_md"] == first["stats"]["record_count"]
    # Component diversity beyond Stack/Card/TextContent/Button
    hist = first["stats"]["component_histogram"]
    assert hist.get("Tabs", 0) >= 1
    assert hist.get("Callout", 0) >= 1
    assert hist.get("Form", 0) >= 1


def test_assess_record_rejects_empty_prompt() -> None:
    record = ExampleRecord(
        id="x",
        prompt="x",
        openui='root = Stack([c])\nc = Button(":c.label")',
        placeholders=[":c.label"],
        split="train",
        design_md="# Design\n",
    )
    report = assess_record(record, require_design_md=False)
    assert "prompt_too_short" in report.reasons
