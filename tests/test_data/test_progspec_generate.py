"""Typed ProgramSpec generation and adaptive coverage invariants."""

from __future__ import annotations

import pytest

from slm_training.data.progspec import emit_record
from slm_training.data.progspec.generate import (
    PROGRAM_FAMILY,
    ComponentCall,
    GeneratorConfig,
    ProgramGenerator,
    Reference,
    TypedProgram,
    TypedStatement,
    generate_program_specs,
)
from slm_training.data.verify import Tier, VerificationContext, verify_record
from slm_training.dsl.language_contract import contract_id
from slm_training.dsl.parser import validate

SMALL_CONFIG = GeneratorConfig(
    components=("TextContent", "Button", "Separator"),
    max_depth=3,
    max_width=3,
    viewports=("mobile", "desktop"),
    render_states=("empty", "populated"),
)


@pytest.fixture(scope="module")
def small_result():
    return ProgramGenerator(SMALL_CONFIG, seed=11).generate_until_covered()


def test_typed_nodes_serialize_without_cfg_sampling() -> None:
    program = TypedProgram(
        root=ComponentCall("Stack", ((Reference("cta"),), "row")),
        statements=(
            TypedStatement(
                "cta",
                ComponentCall("Button", (":cta.label", None, "primary")),
            ),
        ),
    )
    source = program.serialize()
    assert source == (
        'root = Stack([cta], "row")\ncta = Button(":cta.label", null, "primary")'
    )
    assert validate(source).root


def test_small_grid_reaches_full_pairwise_and_selected_triple_coverage(
    small_result,
) -> None:
    assert small_result.coverage["complete"] is True
    assert small_result.coverage["uncovered"] == []
    assert len(small_result.programs) < 25  # no Cartesian product

    axes = small_result.coverage["axes"]
    assert axes["component_pair"]["covered"] == axes["component_pair"]["total"] == 3
    assert axes["component_triple"]["covered"] == 1
    assert axes["prop"]["uncovered"] == []
    assert axes["prop_value_class"]["uncovered"] == []
    assert axes["depth"]["uncovered"] == []
    assert axes["width"]["uncovered"] == []
    assert axes["length"]["uncovered"] == []
    assert axes["viewport_state"]["uncovered"] == []
    assert axes["content_class"]["uncovered"] == []


def test_deferred_contract_axes_are_explicit_not_generated(small_result) -> None:
    unsupported = set(small_result.coverage["unsupported"])
    assert {
        "dataflow:state",
        "dataflow:query",
        "dataflow:mutation",
        "dataflow:action",
        "dataflow:tool",
        "prop:Button.action",
    } <= unsupported
    assert not any(
        token in spec.canonical_openui
        for spec in small_result.programs
        for token in ("$state", "Query(", "Mutation(", "Action(", "@")
    )


def test_every_generated_root_is_split_safe_and_f2_silver(small_result) -> None:
    assert small_result.coverage["verifier"] == {
        "passed": len(small_result.programs),
        "failed": 0,
    }
    assert len({spec.id for spec in small_result.programs}) == len(
        small_result.programs
    )
    for spec in small_result.programs:
        assert spec.contract_id == contract_id()
        assert spec.split == "train"
        digest = spec.id.removeprefix("program_")
        assert digest in spec.lineage_id
        family = spec.program_family_id.removeprefix("generated_")
        assert family in spec.split_group_id
        assert spec.provenance["generator"] == "typed_ast"
        assert spec.facts["depth"] <= SMALL_CONFIG.max_depth
        assert spec.facts["width"] <= SMALL_CONFIG.max_width
        assert spec.facts["content_class"] in SMALL_CONFIG.content_classes
        record = emit_record(
            spec,
            prompt="Generate this typed program.",
            task="generation",
            source=PROGRAM_FAMILY,
        )
        report = verify_record(record, VerificationContext(source_kind="program"))
        assert report.ok
        assert report.tier is Tier.SILVER


def test_generation_is_seed_deterministic() -> None:
    first = generate_program_specs(8, config=SMALL_CONFIG, seed=23)
    second = generate_program_specs(8, config=SMALL_CONFIG, seed=23)
    assert [spec.to_dict() for spec in first.programs] == [
        spec.to_dict() for spec in second.programs
    ]
    assert first.coverage == second.coverage


def test_default_grid_covers_all_published_components_early() -> None:
    result = ProgramGenerator(seed=19).generate(18)
    component_axis = result.coverage["axes"]["component"]
    assert component_axis == {
        "total": 54,
        "covered": 54,
        "uncovered": [],
        "unsupported": [],
    }
    assert result.coverage["verifier"] == {"passed": 18, "failed": 0}


def test_partial_run_reports_uncovered_cells_and_rejects_bad_config() -> None:
    result = generate_program_specs(1, config=SMALL_CONFIG, seed=3)
    assert result.coverage["complete"] is False
    assert result.coverage["uncovered"]
    with pytest.raises(ValueError, match="unknown component"):
        ProgramGenerator(GeneratorConfig(components=("NotAComponent",)))
    with pytest.raises(ValueError, match="positive"):
        GeneratorConfig(max_depth=0)
