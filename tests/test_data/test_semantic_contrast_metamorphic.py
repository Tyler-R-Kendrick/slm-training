"""Regression tests for metamorphic prompt/program generators (SLM-459/VCE-007)."""

from __future__ import annotations

import pytest

from slm_training.data.contract import GenerationRequest, canonical_slot_contract
from slm_training.data.progspec.generate import GeneratorConfig, ProgramGenerator
from slm_training.data.semantic_contrast.builder import _prompt_for
from slm_training.data.semantic_contrast.metamorphic import (
    generate_alpha_rename_case,
    generate_ast_rewrite_equivalence_case,
    generate_prompt_paraphrase_case,
    generate_prompt_single_fact_edit_case,
    generate_reorder_case,
    root_family_for,
)
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.dsl.pack import get_pack
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.meaningful_program import binding_aware_meaningful_v2
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1


@pytest.fixture
def pack():
    return get_pack("openui")


@pytest.fixture
def sample_specs(pack):
    generator = ProgramGenerator(
        GeneratorConfig(
            max_depth=2, max_width=3, components=("TextContent", "Button"), split="train"
        ),
        seed=3,
    )
    return generator.generate(6).programs


def _verdict(pack, openui: str, prompt: str) -> bool:
    record = ExampleRecord(
        id="probe", prompt=prompt, openui=openui, placeholders=[], split="train", source="y"
    )
    request = GenerationRequest(prompt=prompt, slot_contract=canonical_slot_contract(openui))
    return bool(binding_aware_meaningful_v2(openui, record=record, request=request).verdict)


# --------------------------------------------------------------------------
# Alpha-rename
# --------------------------------------------------------------------------


def test_alpha_rename_rejects_gold_provenance_plans(pack, sample_specs):
    extractor = OpenUISemanticPlanExtractor()
    plan = extractor.extract(sample_specs[0], pack)
    assert plan.identity.provenance == "gold"
    assert generate_alpha_rename_case(plan, pack) is None


def test_alpha_rename_leaves_surface_and_fingerprints_unchanged(pack, sample_specs):
    extractor = OpenUISemanticPlanExtractor()
    found_case = False
    for spec in sample_specs:
        plan = extractor.extract(spec, pack)
        predicted = plan.model_copy(
            update={"identity": plan.identity.model_copy(update={"provenance": "predicted"})}
        )
        case = generate_alpha_rename_case(predicted, pack)
        if case is None:
            continue
        found_case = True
        assert case.expected_changed == ()
        assert case.meta["before_seed"] == case.meta["after_seed"]
        assert case.meta["before_fingerprints"] == case.meta["after_fingerprints"]
        assert case.meta["role_map"] or case.meta["symbol_map"]
        # The rename actually changed at least one id string (not a no-op map).
        assert any(old != new for old, new in case.meta["role_map"].items()) or any(
            old != new for old, new in case.meta["symbol_map"].items()
        )
        prompt = _prompt_for(spec)
        assert _verdict(pack, case.meta["before_seed"], prompt) == _verdict(
            pack, case.meta["after_seed"], prompt
        )
    assert found_case, "expected at least one alpha-rename case from the sample sources"


# --------------------------------------------------------------------------
# Reorder declarations
# --------------------------------------------------------------------------


def test_reorder_case_changes_surface_but_preserves_verdict(pack, sample_specs):
    extractor = OpenUISemanticPlanExtractor()
    found_case = False
    for spec in sample_specs:
        plan = extractor.extract(spec, pack)
        case = generate_reorder_case(plan, pack)
        if case is None:
            continue
        found_case = True
        assert case.meta["before_seed"] != case.meta["after_seed"]
        prompt = _prompt_for(spec)
        before_verdict = _verdict(pack, case.meta["before_seed"], prompt)
        after_verdict = _verdict(pack, case.meta["after_seed"], prompt)
        assert before_verdict == after_verdict
        assert before_verdict is True
    assert found_case, "expected at least one reorder case from the sample sources"


# --------------------------------------------------------------------------
# AST rewrite equivalence
# --------------------------------------------------------------------------


def test_ast_rewrite_equivalence_preserves_verdict(sample_specs):
    found_case = False
    for spec in sample_specs:
        case = generate_ast_rewrite_equivalence_case(spec.canonical_openui)
        if case is None:
            continue
        found_case = True
        assert case.meta["semantic_fingerprint_before"] == case.meta["semantic_fingerprint_after"]
        assert case.meta["before"] != case.meta["after"]
        prompt = "Generate this typed OpenUI program."
        pack = get_pack("openui")
        assert _verdict(pack, case.meta["before"], prompt) == _verdict(
            pack, case.meta["after"], prompt
        )
    assert found_case, "expected at least one AST rewrite case from the sample sources"


# --------------------------------------------------------------------------
# Prompt single-fact edit
# --------------------------------------------------------------------------


def test_prompt_single_fact_edit_isolates_the_edited_component():
    base = "Generate an OpenUI program with components: button, text content."
    edited = "Generate an OpenUI program with components: text content."
    case = generate_prompt_single_fact_edit_case(base, edited, edited_component="Button")
    assert case.meta["changed_facts"]
    assert case.meta["changed_mentions_only_target"]
    assert case.meta["base_fact_fingerprints"] != case.meta["edited_fact_fingerprints"]


def test_prompt_single_fact_edit_no_change_yields_no_changed_facts():
    base = "Generate an OpenUI program with components: button, text content."
    case = generate_prompt_single_fact_edit_case(base, base, edited_component="Button")
    assert case.meta["changed_facts"] == []
    assert case.meta["base_fact_fingerprints"] == case.meta["edited_fact_fingerprints"]


# --------------------------------------------------------------------------
# Prompt paraphrase invariance
# --------------------------------------------------------------------------


def test_prompt_paraphrase_invariant_to_whitespace_and_order():
    base = "Generate an OpenUI program with components: button, text content."
    paraphrase = "  Generate  an OpenUI program with components: text content, button.  "
    case = generate_prompt_paraphrase_case(base, paraphrase)
    assert (
        case.meta["base_fact_fingerprints"]["fact_set"]
        == case.meta["paraphrase_fact_fingerprints"]["fact_set"]
    )


# --------------------------------------------------------------------------
# Split lineage
# --------------------------------------------------------------------------


def test_root_family_matches_split_policy_and_is_stable():
    family_a = root_family_for("root = Stack([])", "openui")
    family_b = root_family_for("root = Stack([])", "openui")
    family_c = root_family_for("root = Stack([textcontent1])", "openui")
    assert family_a == family_b
    assert family_a != family_c
    policy = RootFamilySplitPolicyV1()
    assert policy.assign(family_a) in ("train", "validation", "test")
    # require_inherited fails closed on a mismatched split_group_id.
    with pytest.raises(ValueError):
        policy.require_inherited(
            root_family_id=family_a, split_group_id="not-the-family", split="train"
        )


def test_metamorphic_cases_declare_a_root_family_and_split(pack, sample_specs):
    extractor = OpenUISemanticPlanExtractor()
    plan = extractor.extract(sample_specs[0], pack)
    predicted = plan.model_copy(
        update={"identity": plan.identity.model_copy(update={"provenance": "predicted"})}
    )
    case = generate_alpha_rename_case(predicted, pack)
    assert case is not None
    assert case.root_family_id
    assert case.split in ("train", "validation", "test")
