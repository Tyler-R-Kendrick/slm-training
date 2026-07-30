"""Typed ProgramSpec generation and adaptive coverage invariants."""

from __future__ import annotations

import json

import pytest

from scripts.generate_progspecs import main as generate_main
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
from slm_training.dsl.placeholders import extract_placeholders

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


def test_required_component_groups_include_ordered_topology_variants() -> None:
    config = GeneratorConfig(
        components=("TextContent", "Form", "Input"),
        required_components=("TextContent", "Form", "Input"),
        max_width=3,
    )
    generator = ProgramGenerator(config, seed=7)
    orders = {candidate.components for candidate in generator._candidates}

    assert len(orders) == 6
    assert all(set(config.required_components) <= set(order) for order in orders)


def test_required_component_groups_include_configured_supersets() -> None:
    config = GeneratorConfig(
        components=("TextContent", "Form", "Input", "Card"),
        required_components=("TextContent", "Form", "Input"),
        max_width=4,
    )
    generator = ProgramGenerator(config, seed=7)

    orders = {
        candidate.components
        for candidate in generator._candidates
        if len(candidate.components) == 4
    }

    assert len(orders) == 24
    assert all(set(config.required_components) <= set(order) for order in orders)


def test_required_card_topology_contains_the_required_components() -> None:
    config = GeneratorConfig(
        components=("TextContent", "Form", "Input", "Card"),
        required_components=("TextContent", "Form", "Input", "Card"),
        max_width=4,
    )
    result = ProgramGenerator(config, seed=1284).generate(1)

    openui = result.programs[0].canonical_openui
    assert "Card([" in openui
    assert "Card([])" not in openui
    assert "Input(" in openui
    assert "Form(" in openui


def test_required_switch_topology_covers_content_and_nonempty_items() -> None:
    config = GeneratorConfig(
        components=("Slider", "SwitchGroup", "SwitchItem", "TextCallout"),
        required_components=("Slider", "SwitchGroup", "SwitchItem", "TextCallout"),
        max_width=4,
    )
    generator = ProgramGenerator(config, seed=1295)
    targets = {
        (
            candidate.prop_target.component,
            candidate.prop_target.prop,
            candidate.prop_target.variant,
        )
        for candidate in generator._candidates
        if candidate.prop_target is not None
    }

    assert ("SwitchGroup", "items", "nonempty") in targets
    assert ("Slider", "label", "placeholder") in targets
    assert ("TextCallout", "title", "placeholder") in targets

    programs = generator.generate(96).programs
    sources = [program.canonical_openui for program in programs]
    assert all(
        "SwitchGroup(" in source and "[switchitem" in source for source in sources
    )
    assert all(len(extract_placeholders(source)) >= 3 for source in sources)
    assert all(":gen" not in source for source in sources)


def test_required_content_properties_force_the_direct_settings_slot_shape() -> None:
    config = GeneratorConfig(
        components=("SwitchItem", "Slider"),
        required_components=("SwitchItem", "Slider"),
        required_content_properties=(
            ("SwitchItem", "label"),
            ("SwitchItem", "description"),
            ("Slider", "label"),
        ),
        max_width=2,
    )
    programs = ProgramGenerator(config, seed=1302).generate(60).programs

    sources = [program.canonical_openui for program in programs]
    assert all("SwitchGroup(" not in source for source in sources)
    assert all('SwitchItem(":slot_' in source for source in sources)
    assert all('Slider("' in source and ":slot_" in source for source in sources)
    assert all(len(extract_placeholders(source)) == 3 for source in sources)


def test_required_form_topology_resolves_anyof_input_refs() -> None:
    config = GeneratorConfig(
        components=("Form", "FormControl", "Input", "Button"),
        required_components=("Form", "FormControl", "Input", "Button"),
        required_content_properties=(
            ("FormControl", "label"),
            ("Input", "placeholder"),
            ("Button", "label"),
        ),
        max_width=4,
    )
    programs = ProgramGenerator(config, seed=1305).generate(24).programs

    arities = set()
    for program in programs:
        source = program.canonical_openui
        assert "Form(" in source
        arity = source.count("FormControl(")
        arities.add(arity)
        assert source.count("Input(") == arity
        assert source.count("Button(") == arity
        assert validate(source).root
    assert arities == {1, 2, 3}


def test_forward_reference_patterns_pair_slot_ownership_with_slot_free_sharing() -> None:
    config = GeneratorConfig(
        components=("Stack", "Input", "TextContent"),
        forward_reference_patterns=(
            "root_distinct_slot_inputs",
            "root_shared_slot_free_input",
        ),
    )
    result = ProgramGenerator(config, seed=1421).generate_until_covered()

    by_pattern = {
        spec.facts["forward_reference_pattern"]: spec.canonical_openui
        for spec in result.programs
        if "forward_reference_pattern" in spec.facts
    }
    assert set(by_pattern) == set(config.forward_reference_patterns)
    distinct = by_pattern["root_distinct_slot_inputs"]
    shared = by_pattern["root_shared_slot_free_input"]
    assert "root = Stack([input1, input2]" in distinct
    assert len(extract_placeholders(distinct)) == 2
    assert "root = Stack([input1, input1, textcontent2, textcontent3]" in shared
    assert 'input1 = Input("$0", null' in shared
    assert len(extract_placeholders(shared)) == 2
    assert validate(distinct).root and validate(shared).root


def test_required_content_properties_reject_enum_or_unknown_properties() -> None:
    with pytest.raises(ValueError, match="free-form string"):
        ProgramGenerator(
            GeneratorConfig(
                components=("Slider",),
                required_content_properties=(("Slider", "variant"),),
            )
        )
    with pytest.raises(ValueError, match="free-form string"):
        ProgramGenerator(
            GeneratorConfig(
                components=("Slider",),
                required_content_properties=(("Slider", "missing"),),
            )
        )


def test_cli_writes_programs_and_authoritative_coverage(tmp_path) -> None:
    programs_path = tmp_path / "programs.jsonl"
    coverage_path = tmp_path / "coverage.json"
    assert (
        generate_main(
            [
                "--count",
                "2",
                "--seed",
                "7",
                "--output",
                str(programs_path),
                "--coverage",
                str(coverage_path),
            ]
        )
        == 0
    )
    programs = [json.loads(line) for line in programs_path.read_text().splitlines()]
    coverage = json.loads(coverage_path.read_text())
    assert len(programs) == 2
    assert all(
        program["provenance"]["generator"] == "typed_ast" for program in programs
    )
    assert coverage["axes"]["component"]["total"] == 54
    assert coverage["verifier"] == {"failed": 0, "passed": 2}


def test_cli_accepts_required_component_topology_inventory(tmp_path) -> None:
    programs_path = tmp_path / "programs.jsonl"
    coverage_path = tmp_path / "coverage.json"

    assert (
        generate_main(
            [
                "--count",
                "2",
                "--output",
                str(programs_path),
                "--coverage",
                str(coverage_path),
                "--components",
                "TextContent,Form,Input",
                "--required-components",
                "TextContent,Form,Input",
                "--max-width",
                "3",
            ]
        )
        == 0
    )
