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
    records = load_jsonl("src/slm_training/resources/train_seeds.jsonl")
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
        seed_path=Path("src/slm_training/resources/train_seeds.jsonl"),
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


def test_independent_judge_rejects_unrelated_boolean_layout() -> None:
    record = ExampleRecord(
        id="judge-boolean",
        prompt="Emit the OpenUI construct: a boolean literal.",
        openui='root = Stack([sep, cap])\nsep = Separator("horizontal", true)\ncap = TextContent(":cap")',
        placeholders=[":cap"],
    )
    report = assess_record(record, require_design_md=False)
    assert not report.ok
    assert "prompt_lexical_target_wrapped_in_unrelated_layout" in report.reasons


def test_independent_judge_requires_named_component() -> None:
    record = ExampleRecord(
        id="judge-component",
        prompt="Emit the Accordion component.",
        openui='root = Stack([card])\ncard = Card([TextContent(":x")])',
        placeholders=[":x"],
    )
    report = assess_record(record, require_design_md=False)
    assert not report.ok
    assert "prompt_component_missing_from_output" in report.reasons


def test_independent_judge_reads_ordinary_component_mentions_from_schema(
    monkeypatch,
) -> None:
    from slm_training.data import quality

    monkeypatch.setattr(
        quality,
        "_official_component_names",
        lambda: frozenset({"Button", "Buttons", "Callout", "TextContent"}),
    )
    record = ExampleRecord(
        id="judge-prose",
        prompt="Show an informational callout with a title.",
        openui='root = Stack([text])\ntext = TextContent(":title")',
        placeholders=[":title"],
    )
    report = assess_record(record, require_design_md=False)
    assert not report.ok
    assert "prompt_component_missing_from_output" in report.reasons

    plural_record = ExampleRecord(
        id="judge-plural",
        prompt="Show two buttons.",
        openui='root = Stack([a, b])\na = Button(":a")\nb = Button(":b")',
        placeholders=[":a", ":b"],
    )
    assert quality.independent_judge(plural_record)["ok"]

    edit_record = ExampleRecord(
        id="judge-embedded-program",
        prompt=(
            "Current program:\n"
            'root = Callout(\":title\", [label])\nlabel = Label(\":label\")'
        ),
        openui='root = TextContent(":title")',
        placeholders=[":title"],
        meta={"edit": {"instruction": "Update the title copy."}},
    )
    assert quality.independent_judge(edit_record)["ok"]

    repair_record = ExampleRecord(
        id="judge-repair-ast",
        prompt='Repair this program: root = Button(":label")',
        openui='root = TextContent(":title")',
        placeholders=[":title"],
        meta={"repair": {"clean_ast": {"typeName": "Button"}}},
    )
    assert not quality.independent_judge(repair_record)["ok"]


def test_independent_judge_rejects_under_specified_contract_prompt() -> None:
    record = ExampleRecord(
        id="judge-contract",
        prompt="Emit the OpenUI construct: the Button component.",
        openui='root = Stack([button, cap])\nbutton = Button(":label")\ncap = TextContent(":caption")',
        placeholders=[":label", ":caption"],
        meta={"source_family": "language_contract"},
    )
    report = assess_record(record, require_design_md=False)
    assert not report.ok
    assert "prompt_under_specified_for_layout" in report.reasons


def test_normalized_record_stamps_independent_judge_gate() -> None:
    from slm_training.harnesses.train_data.pipeline import _normalize_record

    record = ExampleRecord(
        id="judge-stamp",
        prompt="Emit the OpenUI construct: a Button component.",
        openui='root = Stack([button])\nbutton = Button(":label")',
        placeholders=[":label"],
        split="train",
        design_md="# Design\n",
    )
    stamped = _normalize_record(record)
    gates = stamped.meta["verification"]["gates"]
    judge_gate = next(gate for gate in gates if gate["name"] == "independent_judge")
    assert judge_gate["status"] == "pass"
    assert stamped.meta["independent_judge_passed"] is True
