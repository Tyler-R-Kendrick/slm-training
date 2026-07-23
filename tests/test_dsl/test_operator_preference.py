from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProofV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    CanonicalAstCostV1,
    CompilerCoverage,
    OperatorApplicationV1,
    OperatorPreferenceSequenceV1,
    OperatorPreferenceStepV1,
    PreferenceIntent,
    PreferenceScopeError,
    SemanticPreferenceCandidateV1,
    SequenceDefectPolicy,
    build_operator_preference_group,
    preference_cost,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


FRAME = _sha("semantic-frame")
EQUIVALENCE = _sha("equivalence-class")


def _step(
    label: str,
    *,
    before_state: str = "state-0",
    after_state: str = "state-1",
    before_ast: str = "ast-0",
    after_ast: str = "ast-1",
    action: str | None = None,
    locality: int = 0,
    cost_micros: int = 1_000_000,
    redundant: bool = False,
) -> OperatorPreferenceStepV1:
    return OperatorPreferenceStepV1(
        operator_id=f"openui.{label}",
        operator_fingerprint=_sha(f"operator-{label}"),
        semantic_action_id=_sha(f"action-{action or label}"),
        application_id=_sha(f"application-{label}-{before_state}-{after_state}"),
        before_state_fingerprint=_sha(before_state),
        after_state_fingerprint=_sha(after_state),
        before_ast_fingerprint=_sha(before_ast),
        after_ast_fingerprint=_sha(after_ast),
        locality_violations=locality,
        operator_cost_micros=cost_micros,
        proven_redundant=redundant,
    )


def _sequence(*steps: OperatorPreferenceStepV1) -> OperatorPreferenceSequenceV1:
    if steps:
        initial_state = steps[0].before_state_fingerprint
        initial_ast = steps[0].before_ast_fingerprint
    else:
        initial_state = _sha("state-0")
        initial_ast = _sha("ast-0")
    return OperatorPreferenceSequenceV1(initial_state, initial_ast, tuple(steps))


def _candidate(
    candidate_id: str,
    *,
    nodes: int = 3,
    productions: int | None = None,
    optional_nodes: int = 0,
    markers: int = 0,
    sequence: OperatorPreferenceSequenceV1 | None = None,
    complete: bool = True,
    valid: bool = True,
    frame: str = FRAME,
    equivalence: str = EQUIVALENCE,
) -> SemanticPreferenceCandidateV1:
    sequence = sequence or _sequence(_step(candidate_id))
    return SemanticPreferenceCandidateV1(
        candidate_id=candidate_id,
        semantic_frame_fingerprint=frame,
        equivalence_class_fingerprint=equivalence,
        final_ast_fingerprint=sequence.final_ast_fingerprint,
        required_obligations=("role.hero", "role.action"),
        satisfied_obligations=(
            ("role.hero", "role.action") if complete else ("role.hero",)
        ),
        verifier_valid=valid,
        ast_cost=CanonicalAstCostV1(
            node_count=nodes,
            production_count=productions if productions is not None else nodes * 2,
            optional_node_count=optional_nodes,
            marker_count=markers,
        ),
        sequence=sequence,
    )


def _ranked_ids(group) -> tuple[tuple[str, ...], ...]:
    return tuple(tier.candidate_ids for tier in group.ranked_tiers)


def test_complete_valid_candidate_always_outranks_shorter_incomplete_or_invalid() -> None:
    complete = _candidate("complete", nodes=100)
    incomplete = _candidate("incomplete", nodes=1, complete=False)
    invalid = _candidate("invalid", nodes=0, valid=False)
    group = build_operator_preference_group(
        (invalid, incomplete, complete),
        intent=PreferenceIntent.SIMPLIFY,
    )
    assert _ranked_ids(group)[0] == ("complete",)
    assert group.preference_pairs[0].first_differing_cost_axis == "eligibility"
    rejected_by_complete = {
        pair.rejected_candidate_id
        for pair in group.preference_pairs
        if pair.chosen_candidate_id == "complete"
    }
    assert rejected_by_complete == {"incomplete", "invalid"}


def test_comparison_fails_closed_across_frame_or_equivalence_scope() -> None:
    candidate = _candidate("first")
    for other in (
        replace(_candidate("second"), semantic_frame_fingerprint=_sha("other-frame")),
        replace(
            _candidate("second"),
            equivalence_class_fingerprint=_sha("other-class"),
        ),
    ):
        with pytest.raises(PreferenceScopeError, match="SemanticFrame"):
            build_operator_preference_group(
                (candidate, other), intent=PreferenceIntent.SIMPLIFY
            )


def test_equivalent_outcomes_rank_deterministically_under_input_permutation() -> None:
    candidates = (
        _candidate("large", nodes=7),
        _candidate("small", nodes=2),
        _candidate("middle", nodes=4),
    )
    forward = build_operator_preference_group(
        candidates, intent=PreferenceIntent.SIMPLIFY
    )
    reverse = build_operator_preference_group(
        reversed(candidates), intent=PreferenceIntent.SIMPLIFY
    )
    assert forward.to_dict() == reverse.to_dict()
    assert _ranked_ids(forward) == (("small",), ("middle",), ("large",))


def test_simplification_and_explicit_expansion_reverse_only_structural_cost() -> None:
    shared = _sequence(_step("shared"))
    compact = _candidate("compact", nodes=2, productions=3, sequence=shared)
    expanded = _candidate(
        "expanded",
        nodes=8,
        productions=14,
        optional_nodes=2,
        markers=3,
        sequence=shared,
    )
    simplify = build_operator_preference_group(
        (expanded, compact), intent=PreferenceIntent.SIMPLIFY
    )
    expansion = build_operator_preference_group(
        (compact, expanded), intent=PreferenceIntent.EXPAND
    )
    preserve = build_operator_preference_group(
        (expanded, compact), intent=PreferenceIntent.PRESERVE
    )
    assert _ranked_ids(simplify)[0] == ("compact",)
    assert _ranked_ids(expansion)[0] == ("expanded",)
    assert _ranked_ids(preserve) == (("compact", "expanded"),)
    assert preserve.preference_pairs == ()


def test_no_op_cycle_and_proven_redundancy_are_explicitly_rejected() -> None:
    clean = _candidate("clean")
    no_op = _candidate(
        "noop",
        sequence=_sequence(
            _step(
                "noop",
                before_state="state-0",
                after_state="state-0",
                before_ast="ast-0",
                after_ast="ast-0",
            )
        ),
    )
    cycle = _candidate(
        "cycle",
        sequence=_sequence(
            _step("forward"),
            _step(
                "back",
                before_state="state-1",
                after_state="state-0",
                before_ast="ast-1",
                after_ast="ast-0",
            ),
        ),
    )
    redundant = _candidate(
        "redundant",
        sequence=_sequence(_step("redundant", redundant=True)),
    )
    group = build_operator_preference_group(
        (cycle, clean, no_op, redundant),
        intent=PreferenceIntent.SIMPLIFY,
    )
    assert group.rejected_candidate_ids == ("cycle", "noop", "redundant")
    assert _ranked_ids(group) == (("clean",),)
    assert no_op.sequence.diagnostics.no_op_step_indices == (0,)
    assert cycle.sequence.diagnostics.cycle_step_indices == (1,)
    assert redundant.sequence.diagnostics.redundant_step_indices == (0,)
    with pytest.raises(ValueError, match="rejected by policy"):
        preference_cost(
            no_op,
            intent=PreferenceIntent.SIMPLIFY,
            sequence_defect_policy=SequenceDefectPolicy.REJECT,
        )


def test_penalized_sequence_defects_cannot_win_with_smaller_ast() -> None:
    clean = _candidate("clean", nodes=50)
    no_op = _candidate(
        "noop",
        nodes=0,
        sequence=_sequence(
            _step(
                "noop",
                before_state="state-0",
                after_state="state-0",
                before_ast="ast-0",
                after_ast="ast-0",
            )
        ),
    )
    group = build_operator_preference_group(
        (no_op, clean),
        intent=PreferenceIntent.SIMPLIFY,
        sequence_defect_policy=SequenceDefectPolicy.PENALIZE,
    )
    assert _ranked_ids(group)[0] == ("clean",)
    assert group.preference_pairs[0].first_differing_cost_axis == "sequence_defects"


def test_locality_length_and_operator_cost_are_lexicographic_not_weighted() -> None:
    low_locality = _candidate(
        "lowlocality",
        sequence=_sequence(_step("lowlocality", locality=0, cost_micros=9_000_000)),
    )
    low_operator_cost = _candidate(
        "lowcost",
        sequence=_sequence(_step("lowcost", locality=1, cost_micros=1)),
    )
    group = build_operator_preference_group(
        (low_operator_cost, low_locality),
        intent=PreferenceIntent.PRESERVE,
    )
    assert _ranked_ids(group)[0] == ("lowlocality",)
    assert group.preference_pairs[0].first_differing_cost_axis == (
        "locality_violations"
    )


def test_final_ast_and_semantic_sequence_equivalence_groups_are_materialized() -> None:
    shared_sequence = _sequence(_step("shared", action="same-semantic-action"))
    first = _candidate("first", sequence=shared_sequence)
    second = _candidate("second", sequence=shared_sequence)
    different = _candidate(
        "different",
        sequence=_sequence(
            _step(
                "different",
                action="different-action",
                after_ast="ast-different",
            )
        ),
    )
    group = build_operator_preference_group(
        (different, second, first), intent=PreferenceIntent.PRESERVE
    )
    ast_group = next(
        item
        for item in group.final_ast_groups
        if item.fingerprint == shared_sequence.final_ast_fingerprint
    )
    sequence_group = next(
        item
        for item in group.semantic_sequence_groups
        if item.fingerprint == shared_sequence.semantic_fingerprint
    )
    assert ast_group.candidate_ids == ("first", "second")
    assert sequence_group.candidate_ids == ("first", "second")


def test_equal_costs_form_a_tie_without_fabricated_preference() -> None:
    shared = _sequence(_step("shared"))
    first = _candidate("first", sequence=shared)
    second = _candidate("second", sequence=shared)
    group = build_operator_preference_group(
        (second, first), intent=PreferenceIntent.SIMPLIFY
    )
    assert _ranked_ids(group) == (("first", "second"),)
    assert group.preference_pairs == ()


def test_preference_pair_generation_is_explicitly_bounded() -> None:
    candidates = (
        _candidate("first", nodes=1),
        _candidate("second", nodes=2),
        _candidate("third", nodes=3),
    )
    with pytest.raises(ValueError, match="bound exceeded"):
        build_operator_preference_group(
            candidates,
            intent=PreferenceIntent.SIMPLIFY,
            max_pairs=2,
        )


def test_cost_and_sequence_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CanonicalAstCostV1(-1, 0, 0, 0)
    with pytest.raises(ValueError, match="state-contiguous"):
        _sequence(
            _step("first"),
            _step("second", before_state="wrong-state", before_ast="ast-1"),
        )
    with pytest.raises(ValueError, match="final AST"):
        replace(_candidate("candidate"), final_ast_fingerprint=_sha("wrong"))


def test_step_from_application_binds_declaration_proof_and_exact_cost() -> None:
    declaration = AstOperatorV1(
        operator_id="openui.fixture",
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.25,
    )
    effect = ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT)
    application = OperatorApplicationV1(
        operator_fingerprint=declaration.fingerprint,
        arguments=(),
        before_state_digest=_sha("before-state"),
        before_ast_digest=_sha("before-ast"),
        after_state_digest=_sha("after-state"),
        after_ast_digest=_sha("after-ast"),
        effect=effect,
        proof=ApplicationProofV1(
            proof_kind="compiler.replay",
            checks=("ast.valid",),
            compiler_result_digest=_sha("compiler-result"),
            effect_fingerprint=effect.fingerprint,
        ),
        provenance=ApplicationProvenanceV1(
            pack_id="openui",
            compiler_id="openui.compiler",
            compiler_version="v1",
            source_artifact_digest=_sha("source"),
            request_id="request-1",
        ),
    )
    step = OperatorPreferenceStepV1.from_application(
        declaration=declaration,
        semantic_action_id=_sha("semantic-action"),
        application=application,
    )
    assert step.operator_id == declaration.operator_id
    assert step.operator_cost_micros == 1_250_000
    assert step.after_ast_fingerprint == application.after_ast_digest

    with pytest.raises(ValueError, match="declaration"):
        OperatorPreferenceStepV1.from_application(
            declaration=replace(declaration, operator_id="openui.other"),
            semantic_action_id=_sha("semantic-action"),
            application=application,
        )
