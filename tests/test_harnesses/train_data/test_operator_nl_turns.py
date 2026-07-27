"""DSH3-11 (SLM-379): operator NL turn generation, admission, stop rule."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorArgumentSlotV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    ReferenceEntryV1,
    ReferenceTableV1,
    RegisteredOperatorV1,
    ValueRef,
    build_reference_table,
    deserialize_operator_action,
    enumerate_operator_legal_set,
)
from slm_training.dsl.pack import get_pack
from slm_training.harnesses.train_data.operator_corpus import OperatorTargetView
from slm_training.harnesses.train_data.operator_nl_turns import (
    DISPOSITION_SYMBOLIC_CAP2_ONLY,
    OperatorNlFixtureProviderV1,
    build_operator_disambiguation_report,
    classify_state_references,
    generate_operator_nl_turns,
)

SOURCE = 'root = TextContent(":hero.title")'
OPERATOR_ID = "openui.fixture_nl"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _LeakingProvider(OperatorNlFixtureProviderV1):
    def generate(self, *, system_text, prompt_text, sampling, seed):
        return f"Apply OPERATOR {OPERATOR_ID} to the layout now."


class _MarkerProvider(OperatorNlFixtureProviderV1):
    def generate(self, *, system_text, prompt_text, sampling, seed):
        return "Set the text to :hero.title please."


def _value_descriptors(count: int) -> tuple[ReferenceDescriptorV1, ...]:
    return tuple(
        ReferenceDescriptorV1(
            ref_kind=RefKind.VALUE,
            semantic_fingerprint=_sha(f"semantic-value-{index}"),
            value_type="openui.string",
        )
        for index in range(count)
    )


def _family(
    *,
    count: int,
    allowed: frozenset[int],
    effect_mode: str = "per_arg",
    request_id: str = "request-1",
    duplicate_descriptors: bool = False,
):
    """Build one fixture operator family.

    ``effect_mode="per_arg"``: the effect delta targets the bound argument
    (each legal action has a distinct, non-equivalent effect).
    ``effect_mode="shared"``: the effect delta targets a fixed third ref
    (every legal action has the same effect fingerprint -> equivalent).
    """
    base_pack = get_pack("openui")
    state = OperatorStateV1.from_source(base_pack, SOURCE)
    branch = _sha(f"branch:{request_id}")
    if duplicate_descriptors:
        descriptor = _value_descriptors(1)[0]
        table = ReferenceTableV1(
            request_id=request_id,
            state_digest=state.state_digest,
            branch_digest=branch,
            entries=(
                ReferenceEntryV1(ValueRef(request_id, "refaa"), descriptor),
                ReferenceEntryV1(ValueRef(request_id, "refbb"), descriptor),
            ),
        )
    else:
        table = build_reference_table(
            request_id=request_id,
            state_digest=state.state_digest,
            branch_digest=branch,
            descriptors=_value_descriptors(max(count, 3)),
            seed=7,
        )
    value_refs = sorted(
        (entry.ref for entry in table.entries if entry.ref.KIND is RefKind.VALUE),
        key=lambda ref: ref.opaque_id,
    )
    semantic_by_ref = {
        entry.ref: entry.descriptor.semantic_fingerprint for entry in table.entries
    }
    allowed_semantics = {semantic_by_ref[value_refs[index]] for index in allowed}

    def execute(operator_state, arguments):
        ref = arguments[0].value
        if semantic_by_ref[ref] not in allowed_semantics:
            raise OperatorRejectedError("fixture.not_legal")
        target = ref if effect_mode == "per_arg" else value_refs[2]
        return OperatorMutationV1(
            source=operator_state.source.replace(":hero.title", ":hero.body"),
            effect=ActionEffectV1(
                property_deltas=(
                    EffectDeltaV1(
                        kind=EffectDeltaKind.PROPERTY,
                        target=target,
                        before="draft",
                        after="published",
                    ),
                ),
                compiler_coverage=CompilerCoverage.EXACT,
            ),
        )

    declaration = AstOperatorV1(
        operator_id=OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(EffectDeltaKind.PROPERTY,),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
    pack = replace(base_pack, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(SOURCE),
        request_id=request_id,
    )
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=state,
        reference_table=table,
        provenance=provenance,
    )
    return pack, library, state, table, provenance, legal_set, value_refs


def _apply(family, ref):
    pack, library, state, table, provenance, _, _ = family
    result = library.apply(
        pack, state, OPERATOR_ID, (BoundArgumentV1("value", ref),), provenance
    )
    assert result.succeeded and result.state is not None
    return result


def _generate(family, result, accepted, **kwargs):
    pack, library, state, table, provenance, legal_set, _ = family
    kwargs.setdefault("legal_set", legal_set)
    return generate_operator_nl_turns(
        pack=pack,
        library=library,
        state=state,
        after_state=result.state,
        reference_table=table,
        application=result.application,
        accepted_application_ids=accepted,
        **kwargs,
    )


def test_admission_recovers_accepted_application() -> None:
    family = _family(count=3, allowed=frozenset({0}))
    result = _apply(family, family[6][0])
    turn_set = _generate(family, result, [result.application.application_id])
    assert len(turn_set.rows) == 2  # concise + imperative
    assert turn_set.rejected_rows == ()
    for row in turn_set.rows:
        assert row.target_view == "operator_only"
        operator_id, arguments = deserialize_operator_action(row.target["operator"])
        assert operator_id == OPERATOR_ID
        assert arguments == result.application.arguments
        assert row.state_reference_scoring
        assert all(
            score == "unambiguous" for score in row.state_reference_scoring.values()
        )
        assert row.style_provider_provenance["provider_id"] == (
            "slm379-operator-nl-fixture"
        )


def test_targets_are_symbol_only_across_views() -> None:
    family = _family(count=3, allowed=frozenset({0}))
    pack = family[0]
    result = _apply(family, family[6][0])
    accepted = [result.application.application_id]
    for view in (
        OperatorTargetView.OPERATOR_ONLY,
        OperatorTargetView.RESULT_AST_ONLY,
        OperatorTargetView.DUAL,
    ):
        turn_set = _generate(family, result, accepted, target_view=view)
        assert turn_set.rows
        for row in turn_set.rows:
            assert set(row.target) <= {"operator", "result_ast"}
            assert row.prompt_text not in row.target.values()
            for value in row.target.values():
                assert row.prompt_text not in value
            operator = row.target.get("operator")
            if operator is not None:
                deserialize_operator_action(operator)
            result_ast = row.target.get("result_ast")
            if result_ast is not None:
                # Natural language never enters targets: the result AST is
                # pack-parseable canonical DSL, not prose.
                pack.backend.parse(result_ast)
                assert "make sure" not in result_ast


def test_ambiguous_state_reference_is_rejected() -> None:
    family = _family(count=2, allowed=frozenset({0, 1}), duplicate_descriptors=True)
    result = _apply(family, family[6][0])
    other = family[6][1]
    both = {
        result.application.application_id,
        _apply(family, other).application.application_id,
    }
    turn_set = _generate(family, result, sorted(both))
    assert turn_set.rows == ()
    assert set(turn_set.rejection_counts) == {"ambiguous_state_reference"}


def test_non_equivalent_compatible_prompt_is_rejected() -> None:
    family = _family(count=3, allowed=frozenset({0, 1}))
    result = _apply(family, family[6][0])
    turn_set = _generate(family, result, [result.application.application_id])
    assert turn_set.rows == ()
    assert set(turn_set.rejection_counts) == {"compatible_with_non_equivalent"}


def test_accepted_set_naming_is_admitted() -> None:
    family = _family(count=3, allowed=frozenset({0, 1}))
    result = _apply(family, family[6][0])
    other = _apply(family, family[6][1])
    accepted = sorted(
        {result.application.application_id, other.application.application_id}
    )
    turn_set = _generate(family, result, accepted)
    assert len(turn_set.rows) == 2
    for row in turn_set.rows:
        assert row.names_accepted_set
        assert row.accepted_application_ids == tuple(accepted)
        assert "accepted applications" in row.prompt_text


def test_equivalent_applications_share_one_effect() -> None:
    family = _family(count=3, allowed=frozenset({0, 1}), effect_mode="shared")
    result = _apply(family, family[6][0])
    other = _apply(family, family[6][1])
    accepted = sorted(
        {result.application.application_id, other.application.application_id}
    )
    turn_set = _generate(family, result, accepted)
    assert len(turn_set.rows) == 2
    assert all(row.names_accepted_set for row in turn_set.rows)
    # A single accepted member admits without set naming (the other
    # equivalent application is semantically indistinguishable).
    single = _generate(family, result, [result.application.application_id])
    assert len(single.rows) == 2
    assert all(not row.names_accepted_set for row in single.rows)


def test_missing_legal_membership_is_rejected() -> None:
    family = _family(count=3, allowed=frozenset({0}))
    result = _apply(family, family[6][0])
    other_family = _family(count=3, allowed=frozenset({0}), request_id="request-2")
    turn_set = _generate(
        family,
        result,
        [result.application.application_id],
        legal_set=other_family[5],
    )
    assert turn_set.rows == ()
    assert set(turn_set.rejection_counts) == {"missing_legal_membership"}


def test_not_in_accepted_set_is_rejected() -> None:
    family = _family(count=3, allowed=frozenset({0}))
    result = _apply(family, family[6][0])
    turn_set = _generate(family, result, [_sha("not-the-application")])
    assert turn_set.rows == ()
    assert set(turn_set.rejection_counts) == {"not_in_accepted_set"}


def test_leaking_prompts_are_rejected() -> None:
    family = _family(count=3, allowed=frozenset({0}))
    result = _apply(family, family[6][0])
    accepted = [result.application.application_id]
    for provider in (_LeakingProvider(), _MarkerProvider()):
        turn_set = _generate(family, result, accepted, provider=provider)
        assert turn_set.rows == ()
        assert set(turn_set.rejection_counts) == {"leak"}


def test_state_reference_classifier_scores_shared_descriptors() -> None:
    family = _family(count=3, allowed=frozenset({0}))
    _, _, _, table, _, _, _ = family
    result = _apply(family, family[6][0])
    turn_set = _generate(family, result, [result.application.application_id])
    assert turn_set.rows
    # Direct classifier behavior over a shared-descriptor table.
    shared = _family(count=2, allowed=frozenset({0, 1}), duplicate_descriptors=True)
    shared_result = _apply(shared, shared[6][0])
    frame_scoring = classify_state_references(
        _frame_for(shared, shared_result),
        shared[3],
        equivalent=True,
    )
    assert set(frame_scoring.values()) == {"accepted_set_scored"}
    ambiguous = classify_state_references(
        _frame_for(shared, shared_result),
        shared[3],
        equivalent=False,
    )
    assert set(ambiguous.values()) == {"ambiguous"}


def _frame_for(family, result):
    from slm_training.dsl.operator_frame import derive_operator_frame

    _, library, state, table, _, _, _ = family
    declaration = library.lookup(OPERATOR_ID)
    return derive_operator_frame(
        declaration,
        result.application.arguments,
        state,
        result.application.effect,
        table,
    )


def test_disambiguation_report_records_stop_rule_disposition() -> None:
    good = _family(count=3, allowed=frozenset({0}))
    good_result = _apply(good, good[6][0])
    good_set = _generate(good, good_result, [good_result.application.application_id])
    bad = _family(count=2, allowed=frozenset({0, 1}), duplicate_descriptors=True)
    bad_result = _apply(bad, bad[6][0])
    bad_other = _apply(bad, bad[6][1])
    bad_set = _generate(
        bad,
        bad_result,
        sorted(
            {
                bad_result.application.application_id,
                bad_other.application.application_id,
            }
        ),
    )
    report = build_operator_disambiguation_report(
        {"fixture_unambiguous": good_set, "fixture_ambiguous": bad_set}
    )
    by_family = {family.family: family for family in report.families}
    assert by_family["fixture_unambiguous"].disambiguation_rate == 1.0
    assert by_family["fixture_unambiguous"].disposition == "nl_prompts_admitted"
    assert by_family["fixture_ambiguous"].disambiguation_rate == 0.0
    assert by_family["fixture_ambiguous"].disposition == (
        DISPOSITION_SYMBOLIC_CAP2_ONLY
    )
    assert report.symbolic_cap2_only_families == ("fixture_ambiguous",)
    assert 0.0 < report.overall_rate < 1.0


def test_generation_is_deterministic() -> None:
    family = _family(count=3, allowed=frozenset({0}))
    result = _apply(family, family[6][0])
    accepted = [result.application.application_id]
    clock = lambda: "2026-07-25T00:00:00+00:00"  # noqa: E731
    first = _generate(family, result, accepted, now=clock)
    second = _generate(family, result, accepted, now=clock)
    assert first.to_dict() == second.to_dict()
    assert [row.prompt_sha256 for row in first.rows] == [
        row.prompt_sha256 for row in second.rows
    ]
