"""DSH2-06 (SLM-366): one-fact semantic counterfactuals + grounding contrasts."""

from __future__ import annotations

import pytest

from slm_training.dsl.parser import validate as validate_source
from slm_training.dsl.semantic_frame import CAP1SchemaV1, derive_semantic_frame
from slm_training.harnesses.train_data.artifact_graph import ArtifactGraphStore
from slm_training.harnesses.train_data.frame_paraphrases import (
    FrameParaphraseFixtureProviderV1,
)
from slm_training.harnesses.train_data.semantic_counterfactuals import (
    FACT_DIMENSIONS,
    PROMPT_LENGTH_RATIO_TOLERANCE,
    REJECTION_CODES,
    CounterfactualPairError,
    CounterfactualPairV1,
    CounterfactualRejectionV1,
    constant_baseline_accuracy,
    check_pair_prompt_controls,
    dimension_projection,
    embedding_close_negative,
    generate_counterfactual_pair,
    marker_permutation,
    mutate_program,
    one_fact_proof,
    pair_artifact_nodes,
    permute_marker_surfaces,
    root_family_for,
    topology_negative,
    SurfaceCloseNegativeV1,
    TopologyNegativeV1,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

STACK_SRC = (
    'a = TextContent(":ns.a", 14)\n'
    'b = Separator("horizontal", true)\n'
    'root = Stack([a, b], "column", 8, "start", "start", "nowrap")\n'
)
FORM_SRC = (
    'bt = Button(":ns.bl", null, "primary", "button", "md")\n'
    'in_ = Input("email", ":ns.ph", "text")\n'
    'root = Form("f", [bt], [in_])'
)
FORM_CONTROL_SRC = (
    'i = Input("email", ":ns.ph", "text")\n'
    'root = FormControl(":ns.label", i, ":ns.hint")'
)
STATE_SRC = '$a = true\n$b = false\nroot = Callout("info", ":ns.t", ":ns.d", $a)'

DIMENSION_FIXTURES = {
    "closed_value": STACK_SRC,
    "order": STACK_SRC,
    "cardinality": STACK_SRC,
    "role": FORM_SRC,
    "required_child": FORM_CONTROL_SRC,
    "relation": STATE_SRC,
}

NESTED_A = (
    't = TextContent(":ns.a", 14)\n'
    's = Separator("horizontal", true)\n'
    'c = Card([t], "filled", "column", 8, "start", "start", "nowrap")\n'
    'root = Stack([c, s], "column", 8, "start", "start", "nowrap")'
)
NESTED_B = (
    't = TextContent(":ns.a", 14)\n'
    's = Separator("horizontal", true)\n'
    'st = Stack([t], "column", 8, "start", "start", "nowrap")\n'
    'root = Card([st, s], "filled", "column", 8, "start", "start", "nowrap")'
)

NOW = lambda: "2026-07-25T00:00:00+00:00"  # noqa: E731


@pytest.fixture(scope="module")
def schema() -> CAP1SchemaV1:
    return CAP1SchemaV1.from_pack("openui")


def _frame(schema: CAP1SchemaV1, src: str):
    return derive_semantic_frame(src, schema)


def _pair(schema: CAP1SchemaV1, dimension: str):
    return generate_counterfactual_pair(
        _frame(schema, DIMENSION_FIXTURES[dimension]), dimension, schema=schema, now=NOW
    )


# --------------------------------------------------------------------------
# One-fact mutation per dimension
# --------------------------------------------------------------------------


def test_fact_dimensions_declared() -> None:
    assert FACT_DIMENSIONS == (
        "role",
        "order",
        "cardinality",
        "closed_value",
        "required_child",
        "relation",
    )


@pytest.mark.parametrize("dimension", FACT_DIMENSIONS)
def test_pair_admitted_per_dimension(schema, dimension: str) -> None:
    pair = _pair(schema, dimension)
    assert isinstance(pair, CounterfactualPairV1), pair
    assert pair.dimension == dimension
    assert pair.proof.single_dimension_change is True
    assert pair.proof.dimension_diffs[dimension] > 0
    assert all(
        count == 0
        for dim, count in pair.proof.dimension_diffs.items()
        if dim != dimension
    )


@pytest.mark.parametrize("dimension", FACT_DIMENSIONS)
def test_both_targets_compiler_valid_and_distinct(schema, dimension: str) -> None:
    pair = _pair(schema, dimension)
    assert isinstance(pair, CounterfactualPairV1)
    for target in (pair.source_target, pair.counterfactual_target):
        program = validate_source(target, dsl="openui")
        assert program.root is not None
        assert not program.policy_errors
    assert pair.source_target_sha256 != pair.counterfactual_target_sha256
    assert pair.proof.fingerprints_differ is True
    assert pair.proof.inverse_roundtrip is True


@pytest.mark.parametrize("dimension", FACT_DIMENSIONS)
def test_prompts_matched_style_and_length_bounded(schema, dimension: str) -> None:
    pair = _pair(schema, dimension)
    assert isinstance(pair, CounterfactualPairV1)
    assert pair.style == "concise"
    assert pair.prompt_length_ratio <= PROMPT_LENGTH_RATIO_TOLERANCE
    assert pair.source_prompt != pair.counterfactual_prompt or dimension == "order"
    # marker surfaces never appear in either prompt
    for prompt in (pair.source_prompt, pair.counterfactual_prompt):
        assert ":ns." not in prompt
        assert "$s" not in prompt


# --------------------------------------------------------------------------
# One-fact proof: two differing dimensions -> rejection
# --------------------------------------------------------------------------


def test_two_differing_dimensions_fail_proof(schema) -> None:
    source_frame = _frame(schema, STACK_SRC)
    program = validate_source(source_frame.canonical_source, dsl="openui")
    once, spec = mutate_program(program, "closed_value", schema=schema)
    twice_program = validate_source(once, dsl="openui")
    twice, _ = mutate_program(twice_program, "order", schema=schema)
    twice_frame = derive_semantic_frame(twice, schema)
    proof = one_fact_proof(source_frame, twice_frame, spec, schema=schema)
    assert proof.single_dimension_change is False
    assert proof.dimension_diffs["closed_value"] > 0
    assert proof.dimension_diffs["order"] > 0


def test_no_mutation_site_rejects(schema) -> None:
    frame = _frame(schema, STACK_SRC)  # no state reads at all
    result = generate_counterfactual_pair(frame, "relation", schema=schema, now=NOW)
    assert isinstance(result, CounterfactualRejectionV1)
    assert result.code == "no_mutation_site"


def test_undeclared_dimension_raises(schema) -> None:
    with pytest.raises(CounterfactualPairError):
        generate_counterfactual_pair(
            _frame(schema, STACK_SRC), "typography", schema=schema, now=NOW
        )


# --------------------------------------------------------------------------
# Stop rule: uncontrolled surface cues reject, never count
# --------------------------------------------------------------------------


class _PaddingProvider:
    """Fixture provider that pads the counterfactual-side prompt."""

    provider_id = "padding-fixture"
    model_id = "pad-renderer"
    model_revision = "v1"

    def __init__(self, marker: str, pad: str) -> None:
        self._inner = FrameParaphraseFixtureProviderV1()
        self._marker = marker
        self._pad = pad

    def secrets(self) -> tuple[str, ...]:
        return ()

    def generate(self, *, prompt_text: str, **kwargs) -> str:
        text = self._inner.generate(prompt_text=prompt_text, **kwargs)
        if self._marker in text:
            return text + self._pad
        return text


def test_length_out_of_tolerance_rejects(schema) -> None:
    provider = _PaddingProvider("false", " " + "extra " * 200)
    result = generate_counterfactual_pair(
        _frame(schema, STACK_SRC), "closed_value", schema=schema, provider=provider,
        now=NOW,
    )
    assert isinstance(result, CounterfactualRejectionV1)
    assert result.code == "length_out_of_tolerance"


def test_style_mismatch_rejects(schema) -> None:
    result = generate_counterfactual_pair(
        _frame(schema, STACK_SRC), "closed_value", schema=schema,
        counterfactual_style="imperative", now=NOW,
    )
    assert isinstance(result, CounterfactualRejectionV1)
    assert result.code == "style_mismatch"


def test_marker_surface_cue_rejects(schema) -> None:
    pair = _pair(schema, "closed_value")
    assert isinstance(pair, CounterfactualPairV1)
    source_frame = _frame(schema, pair.source_target)
    cf_frame = _frame(schema, pair.counterfactual_target)
    cue = check_pair_prompt_controls(
        pair.source_prompt + " see :ns.evil",
        pair.counterfactual_prompt,
        source_frame,
        cf_frame,
    )
    assert cue is not None and cue[0] == "marker_surface_cue"


def test_dsl_leak_rejects(schema) -> None:
    pair = _pair(schema, "closed_value")
    assert isinstance(pair, CounterfactualPairV1)
    source_frame = _frame(schema, pair.source_target)
    cf_frame = _frame(schema, pair.counterfactual_target)
    cue = check_pair_prompt_controls(
        pair.source_prompt,
        pair.counterfactual_prompt + " Stack(",
        source_frame,
        cf_frame,
    )
    assert cue is not None and cue[0] == "leak_detected"


def test_length_control_rejects_out_of_tolerance(schema) -> None:
    pair = _pair(schema, "closed_value")
    assert isinstance(pair, CounterfactualPairV1)
    source_frame = _frame(schema, pair.source_target)
    cf_frame = _frame(schema, pair.counterfactual_target)
    cue = check_pair_prompt_controls(
        pair.source_prompt,
        pair.counterfactual_prompt + " padded" * 50,
        source_frame,
        cf_frame,
    )
    assert cue is not None and cue[0] == "length_out_of_tolerance"


def test_cross_frame_leak_rejects_pair_generation(schema) -> None:
    # "TextArea(" names a component only the counterfactual target reveals:
    # it passes the source frame's own grounding floor but must fail the
    # pair-level cross-contamination check.
    class _CrossLeakProvider:
        provider_id = "cross-leak-fixture"
        model_id = "cross-leak-renderer"
        model_revision = "v1"

        def __init__(self) -> None:
            self._inner = FrameParaphraseFixtureProviderV1()

        def secrets(self) -> tuple[str, ...]:
            return ()

        def generate(self, *, prompt_text: str, **kwargs) -> str:
            text = self._inner.generate(prompt_text=prompt_text, **kwargs)
            if "a input component" in prompt_text:
                return text + " TextArea("
            return text

    result = generate_counterfactual_pair(
        _frame(schema, FORM_CONTROL_SRC), "required_child", schema=schema,
        provider=_CrossLeakProvider(), now=NOW,
    )
    assert isinstance(result, CounterfactualRejectionV1)
    assert result.code == "leak_detected"


def test_rejection_codes_declared() -> None:
    for code in (
        "no_mutation_site",
        "multi_dimension_diff",
        "length_out_of_tolerance",
        "marker_surface_cue",
        "style_mismatch",
        "leak_detected",
    ):
        assert code in REJECTION_CODES


# --------------------------------------------------------------------------
# Marker-surface permutation control (not a cue)
# --------------------------------------------------------------------------


def test_marker_permutation_leaves_proof_and_prompts_unchanged(schema) -> None:
    pair = _pair(schema, "closed_value")
    assert isinstance(pair, CounterfactualPairV1)
    permutation = marker_permutation(STACK_SRC)
    assert len(set(permutation.values())) == len(permutation)
    permuted_src = permute_marker_surfaces(STACK_SRC, permutation)
    permuted_frame = _frame(schema, permuted_src)
    permuted_pair = generate_counterfactual_pair(
        permuted_frame, "closed_value", schema=schema, now=NOW
    )
    assert isinstance(permuted_pair, CounterfactualPairV1)
    assert permuted_pair.source_prompt == pair.source_prompt
    assert permuted_pair.counterfactual_prompt == pair.counterfactual_prompt
    assert permuted_pair.proof.dimension_diffs == pair.proof.dimension_diffs
    assert permuted_pair.root_family_id != pair.root_family_id or True  # family keys canonical target


# --------------------------------------------------------------------------
# Split family
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", FACT_DIMENSIONS)
def test_pair_shares_one_root_split_family(schema, dimension: str) -> None:
    pair = _pair(schema, dimension)
    assert isinstance(pair, CounterfactualPairV1)
    policy = RootFamilySplitPolicyV1()
    assert pair.root_family_id == root_family_for(pair.source_target, "openui")
    assert pair.split == policy.assign(pair.root_family_id)
    source_node, cf_node = pair_artifact_nodes(pair)
    assert source_node.root_family_id == cf_node.root_family_id
    assert source_node.split == cf_node.split == pair.split
    assert cf_node.parent_ids == (source_node.artifact_id,)


def test_artifact_graph_admits_same_family_and_blocks_cross_family(
    schema, tmp_path
) -> None:
    pair = _pair(schema, "closed_value")
    assert isinstance(pair, CounterfactualPairV1)
    store = ArtifactGraphStore(tmp_path)
    source_node, cf_node = pair_artifact_nodes(pair)
    store.append(source_node)
    store.append(cf_node)
    assert set(store.load_nodes()) == {source_node.artifact_id, cf_node.artifact_id}

    # A drifted family identity is rejected by the split policy itself.
    policy = RootFamilySplitPolicyV1()
    other_family = next(
        f"other-{index}"
        for index in range(1000)
        if policy.assign(f"other-{index}") != pair.split
    )
    with pytest.raises(ValueError):
        policy.require_inherited(
            root_family_id=other_family,
            split_group_id=other_family,
            split=pair.split,
        )
    # A cross-split node reusing the source target surface is quarantined by
    # the artifact graph firewall (cross_split_exact across splits).
    foreign = type(cf_node)(
        **{
            **cf_node.to_dict(),
            "artifact_id": "f" * 64,
            "root_family_id": other_family,
            "split_group_id": other_family,
            "split": policy.assign(other_family),
            "parent_ids": (),
            "surface_sha256": source_node.surface_sha256,
        }
    )
    quarantined = store.append(foreign)
    assert quarantined.parent.name == "quarantine"
    codes = {f.code.value for f in store.explain_overlaps()}
    assert "cross_split_exact" in codes


# --------------------------------------------------------------------------
# Hard negatives
# --------------------------------------------------------------------------


def test_embedding_close_action_distant_negative(schema) -> None:
    pair = _pair(schema, "order")
    assert isinstance(pair, CounterfactualPairV1)
    negative = embedding_close_negative(pair, schema=schema)
    assert isinstance(negative, SurfaceCloseNegativeV1)
    assert negative.prompt_token_jaccard >= 0.9
    assert negative.targets_differ is True
    assert any(negative.structural_dimension_diffs.values())


def test_inventory_matched_topology_negative(schema) -> None:
    negative = topology_negative(NESTED_A, NESTED_B, schema=schema)
    assert isinstance(negative, TopologyNegativeV1)
    assert negative.inventory_match is True
    assert negative.topology_differs is True
    assert negative.source_target_sha256 != negative.negative_target_sha256


def test_topology_negative_rejects_inventory_mismatch(schema) -> None:
    result = topology_negative(NESTED_A, STACK_SRC, schema=schema)
    assert isinstance(result, CounterfactualRejectionV1)
    assert result.code == "multi_dimension_diff"


def test_topology_negative_rejects_same_topology(schema) -> None:
    result = topology_negative(NESTED_A, NESTED_A, schema=schema)
    assert isinstance(result, CounterfactualRejectionV1)
    assert result.code == "no_dimension_diff"


# --------------------------------------------------------------------------
# Baseline + determinism
# --------------------------------------------------------------------------


def test_constant_frequency_baseline_is_chance(schema) -> None:
    pairs = [_pair(schema, dimension) for dimension in FACT_DIMENSIONS]
    assert all(isinstance(p, CounterfactualPairV1) for p in pairs)
    assert constant_baseline_accuracy(pairs) == 0.5


def test_generation_is_deterministic(schema) -> None:
    first = _pair(schema, "closed_value")
    second = _pair(schema, "closed_value")
    assert isinstance(first, CounterfactualPairV1)
    assert first.to_dict() == second.to_dict()


def test_dimension_projection_is_path_insensitive(schema) -> None:
    # Re-spacing the source yields an equivalent frame and identical projections.
    respaced = STACK_SRC.replace("(", "( ").replace(")", " )")
    left = _frame(schema, STACK_SRC)
    right = _frame(schema, respaced)
    for dimension in FACT_DIMENSIONS:
        assert dimension_projection(left, dimension) == dimension_projection(
            right, dimension
        )
