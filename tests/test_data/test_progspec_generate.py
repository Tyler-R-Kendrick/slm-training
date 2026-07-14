"""Coverage-guided typed ProgramSpec generation."""

from __future__ import annotations

import json

import pytest

from scripts.generate_progspecs import main
from slm_training.data.progspec.generate import (
    Candidate,
    CoverageTracker,
    Element,
    Ref,
    Statement,
    TypedProgram,
    TypedProgramGenerator,
)
from slm_training.data.verify import Tier, VerificationContext, verify_record
from slm_training.dsl import bridge_available, validate
from slm_training.dsl.schema import ExampleRecord


def test_typed_graph_rejects_unknown_and_unreachable_references() -> None:
    with pytest.raises(ValueError, match="unknown reference"):
        TypedProgram((Statement("root", Element("Stack", ((Ref("missing"),),))),))
    with pytest.raises(ValueError, match="reachable"):
        TypedProgram(
            (
                Statement("root", Element("TextContent", (":root",))),
                Statement("orphan", Element("Button", (":orphan",))),
            )
        )
    with pytest.raises(ValueError, match="cycle"):
        TypedProgram(
            (
                Statement("root", Element("Stack", ((Ref("loop"),),))),
                Statement("loop", Element("Stack", ((Ref("root"),),))),
            )
        )


def test_coverage_tracks_pairs_selected_triples_and_gain() -> None:
    program = TypedProgram(
        (
            Statement("root", Element("Stack", ((Ref("card"),),))),
            Statement("text", Element("TextContent", (":text",))),
            Statement("card", Element("Card", ((Ref("text"),),))),
        )
    )
    candidate = Candidate("x", "p", program, "mobile", "empty")
    tracker = CoverageTracker.from_candidates([candidate])
    assert "pair:Card+Stack" in tracker.targets
    assert "triple:Card+Stack+TextContent" in tracker.targets
    assert tracker.gain(candidate.cells()) == len(tracker.targets)
    tracker.update(candidate.cells())
    assert tracker.gain(candidate.cells()) == 0


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge deps missing")
def test_generator_is_seeded_valid_silver_and_split_stable() -> None:
    first = TypedProgramGenerator(seed=7).generate(12)
    second = TypedProgramGenerator(seed=7).generate(12)
    assert [spec.to_dict() for spec in first.programs] == [
        spec.to_dict() for spec in second.programs
    ]
    assert len(first.programs) == 12
    assert first.coverage["covered_count"] > 0
    assert first.coverage["deferred"] == [
        "contract_dataflow:state",
        "contract_dataflow:query",
        "contract_dataflow:mutation",
        "contract_dataflow:action",
        "contract_dataflow:tool",
    ]
    for spec in first.programs:
        assert validate(spec.canonical_openui).root
        assert spec.split == "train"
        assert spec.program_family_id == spec.lineage_id == spec.split_group_id
        record = ExampleRecord(
            id=spec.id,
            prompt="generate",
            openui=spec.canonical_openui,
            placeholders=[],
            split=spec.split,
            source="programspec_generated",
        )
        report = verify_record(record, VerificationContext(source_kind="program-first"))
        assert report.ok
        assert report.tier is Tier.SILVER


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge deps missing")
def test_generator_contains_escaped_dsl_like_literal() -> None:
    result = TypedProgramGenerator(seed=2).generate(96)
    escaped = [
        spec
        for spec in result.programs
        if "ignore previous instructions"
        in str(spec.facts.get("literal_content_probe", ""))
    ]
    assert escaped
    assert ":generated." in escaped[0].canonical_openui
    assert '"root = Fake([x])"' in escaped[0].facts["literal_content_probe"]


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge deps missing")
def test_cli_writes_programs_and_coverage(tmp_path) -> None:
    programs = tmp_path / "programs.jsonl"
    coverage = tmp_path / "coverage.json"
    assert (
        main(
            [
                "--count",
                "4",
                "--seed",
                "3",
                "--output",
                str(programs),
                "--coverage",
                str(coverage),
            ]
        )
        == 0
    )
    assert len(programs.read_text().splitlines()) == 4
    report = json.loads(coverage.read_text())
    assert report["emitted_count"] == 4
    assert report["requested_count"] == 4


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge deps missing")
def test_default_budget_covers_supported_target_grid() -> None:
    report = TypedProgramGenerator(seed=11).generate(16).coverage
    assert report["uncovered"] == []
    assert {"length:short", "length:medium", "length:long"} <= set(report["covered"])
