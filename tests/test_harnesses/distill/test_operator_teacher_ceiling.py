"""Tests for DSH4-02 operator-teacher-ceiling harness (SLM-387).

Exercises the acceptance bar from the Linear issue directly:

* each baseline comparator (frequency, deterministic compiler order, current
  scorer, simple descriptor similarity, oracle accepted set) ranks correctly,
  including missing-score and tied-score edge cases;
* every metric (accepted-set mass, top-k recall, MRR, NDCG, calibration,
  selective risk, verifier-regret correlation) handles empty legal sets,
  single-candidate legal sets, and ties without fabricating a value;
* the synthetic teacher's ranking is provably invariant to candidate-order,
  opaque-id, description-length, and prompt-template perturbation;
* the stop-rule gate returns both a genuine "go" and a genuine "no-go/defer"
  from hand-built evidence, and never partially passes;
* approximate (budget-truncated) decision-state traces carry explicit
  reliability metadata on every ranking, not just the teacher's.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import pytest

from slm_training.dsl.operators.contracts import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorArgumentSlotV1,
    RefKind,
)
from slm_training.dsl.operators.references import ReferenceDescriptorV1, build_reference_table
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RegisteredOperatorV1,
)
from slm_training.dsl.pack import get_pack
from slm_training.evals.solver_state_supervision import SupervisionSource
from slm_training.harnesses.distill.legal_set_teacher_trace import TeacherTraceManifest
from slm_training.harnesses.distill.operator_decision_state import (
    capture_operator_decision_state,
)
from slm_training.harnesses.distill import operator_teacher_ceiling as otc
from slm_training.harnesses.distill import prompted_teacher as pt

OPERATOR_ID = "openui.dsh4_02_test_fixture"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _descriptor(index: int) -> ReferenceDescriptorV1:
    return ReferenceDescriptorV1(
        ref_kind=RefKind.VALUE,
        semantic_fingerprint=_sha(f"dsh4-02-test-candidate-{index}"),
        value_type="openui.level",
    )


def _declaration() -> AstOperatorV1:
    return AstOperatorV1(
        operator_id=OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(EffectDeltaKind.TOPOLOGY,),
        locality="node",
        cost=1.0,
    )


def _executor(semantic_by_ref: dict):
    def execute(state, arguments):
        ref = arguments[0].value
        index = semantic_by_ref[ref]
        return OperatorMutationV1(
            source=state.source.replace(":box.pending", f":box.level{index}"),
            effect=ActionEffectV1(
                topology_deltas=(
                    EffectDeltaV1(
                        EffectDeltaKind.TOPOLOGY, ref, "box.pending", f"box.level{index}"
                    ),
                ),
                compiler_coverage=CompilerCoverage.EXACT,
            ),
        )

    return execute


@dataclass
class _Fixture:
    trace: object
    pack: object
    library: object
    state0: object
    provenance: object
    ref_by_index: dict


def _build(
    *,
    trace_id: str,
    n_candidates: int = 5,
    accepted_indices: tuple[int, ...] = (),
    current_scores_by_index: dict[int, float] | None = None,
    supervision: SupervisionSource = SupervisionSource.GOLD,
    max_combinations_per_operator: int = 10_000,
    approximate: bool = False,
) -> _Fixture:
    base_pack = get_pack("openui")
    source = 'root = TextContent(":box.pending")'
    state0 = OperatorStateV1.from_source(base_pack, source)
    descriptors = tuple(_descriptor(i) for i in range(n_candidates))
    table = build_reference_table(
        request_id=f"request-{trace_id}",
        state_digest=state0.state_digest,
        branch_digest=_sha(f"branch-{trace_id}"),
        descriptors=descriptors,
        seed=1,
    )
    index_by_semantic = {
        _descriptor(i).semantic_fingerprint: i for i in range(n_candidates)
    }
    semantic_by_ref = {
        entry.ref: index_by_semantic[entry.descriptor.semantic_fingerprint]
        for entry in table.entries
    }
    library = OperatorLibraryV1(
        (RegisteredOperatorV1(_declaration(), _executor(semantic_by_ref)),)
    )
    pack = replace(base_pack, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.dsh4_02_test_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(source + trace_id),
        request_id=f"request-{trace_id}",
    )
    manifest = TeacherTraceManifest(
        manifest_id=f"manifest-{trace_id}",
        teacher_model_id="fixture/teacher",
        teacher_revision="fixture",
        prompt_template_hash=_sha("dsh4-02-test-prompt-template"),
        pack_id="openui",
        compiler_version="openui-fixture",
        state_schema_version="dsh4-01.v1",
        timestamp="2026-07-25T00:00:00+00:00",
        provenance={"source": "synthetic"},
    )
    ref_by_index = {
        index_by_semantic[entry.descriptor.semantic_fingerprint]: entry.ref
        for entry in table.entries
    }

    def _app_id(index: int) -> str:
        dry = library.dry_run(
            pack, state0, OPERATOR_ID, (BoundArgumentV1("value", ref_by_index[index]),), provenance
        )
        assert dry.succeeded, dry.rejection
        return dry.application_id

    accepted_ids = tuple(_app_id(i) for i in accepted_indices)
    current_scores = None
    if current_scores_by_index is not None:
        current_scores = {_app_id(i): score for i, score in current_scores_by_index.items()}

    trace = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state0,
        reference_table=table,
        provenance=provenance,
        manifest=manifest,
        supervision_source=supervision,
        trace_id=trace_id,
        accepted_application_ids=accepted_ids,
        current_scores=current_scores,
        approximate=approximate,
        max_combinations_per_operator=max_combinations_per_operator,
    )
    return _Fixture(trace, pack, library, state0, provenance, ref_by_index)


# --------------------------------------------------------------------------- #
# RankingV1
# --------------------------------------------------------------------------- #


def test_ranking_v1_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="must not repeat"):
        otc.RankingV1(
            trace_id="t",
            source_id="s",
            application_order=("a", "a"),
            probabilities=None,
            reliability=otc.RankingReliability.EXACT,
            reliability_reason="r",
            approximate=False,
        )


def test_ranking_v1_rejects_probabilities_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        otc.RankingV1(
            trace_id="t",
            source_id="s",
            application_order=("a", "b"),
            probabilities=(0.2, 0.2),
            reliability=otc.RankingReliability.EXACT,
            reliability_reason="r",
            approximate=False,
        )


def test_ranking_v1_empty_order_cannot_carry_probabilities() -> None:
    with pytest.raises(ValueError, match="empty ranking cannot carry probabilities"):
        otc.RankingV1(
            trace_id="t",
            source_id="s",
            application_order=(),
            probabilities=(1.0,),
            reliability=otc.RankingReliability.EXACT,
            reliability_reason="r",
            approximate=False,
        )


def test_ranking_v1_empty_order_is_valid() -> None:
    ranking = otc.RankingV1(
        trace_id="t",
        source_id="s",
        application_order=(),
        probabilities=None,
        reliability=otc.RankingReliability.EXACT,
        reliability_reason="r",
        approximate=False,
    )
    assert ranking.application_order == ()


# --------------------------------------------------------------------------- #
# Decision families
# --------------------------------------------------------------------------- #


def test_classify_decision_family_buckets_by_size_and_supervision() -> None:
    empty = _build(trace_id="fam-empty", n_candidates=0, supervision=SupervisionSource.GOLD).trace
    singleton = _build(
        trace_id="fam-singleton", n_candidates=1, supervision=SupervisionSource.GOLD
    ).trace
    small = _build(
        trace_id="fam-small", n_candidates=3, supervision=SupervisionSource.ON_POLICY
    ).trace
    large = _build(
        trace_id="fam-large", n_candidates=6, supervision=SupervisionSource.ON_POLICY
    ).trace

    assert otc.classify_decision_family(empty) is otc.DecisionFamily.GOLD_EMPTY
    assert otc.classify_decision_family(singleton) is otc.DecisionFamily.GOLD_SINGLETON
    assert otc.classify_decision_family(small) is otc.DecisionFamily.ON_POLICY_SMALL
    assert otc.classify_decision_family(large) is otc.DecisionFamily.ON_POLICY_LARGE


# --------------------------------------------------------------------------- #
# Baseline comparators
# --------------------------------------------------------------------------- #


def test_compiler_order_baseline_matches_legal_set_enumeration_order() -> None:
    fx = _build(trace_id="cob-1", n_candidates=5)
    ranking = otc.CompilerOrderBaselineV1().rank(fx.trace)
    assert ranking is not None
    assert ranking.application_order == tuple(
        a.application_id for a in fx.trace.legal_set.operator_actions
    )
    assert ranking.probabilities is None
    assert ranking.reliability is otc.RankingReliability.EXACT
    assert ranking.approximate is False


def test_compiler_order_baseline_empty_legal_set() -> None:
    fx = _build(trace_id="cob-empty", n_candidates=0)
    ranking = otc.CompilerOrderBaselineV1().rank(fx.trace)
    assert ranking is not None
    assert ranking.application_order == ()
    assert ranking.probabilities is None


def test_frequency_baseline_learns_a_skewed_training_prior() -> None:
    training = [
        _build(trace_id=f"freq-train-{i}", n_candidates=5, accepted_indices=(2,)).trace
        for i in range(8)
    ]
    table = otc.build_frequency_table(training)
    assert table.total == 8

    eval_fx = _build(trace_id="freq-eval-1", n_candidates=5)
    ranking = otc.FrequencyBaselineV1(table).rank(eval_fx.trace)
    assert ranking is not None
    preferred_id = eval_fx.library.dry_run(
        eval_fx.pack,
        eval_fx.state0,
        OPERATOR_ID,
        (BoundArgumentV1("value", eval_fx.ref_by_index[2]),),
        eval_fx.provenance,
    ).application_id
    assert ranking.application_order[0] == preferred_id
    assert abs(sum(ranking.probabilities) - 1.0) < 1e-9


def test_frequency_baseline_with_no_training_data_ties_and_breaks_by_id() -> None:
    table = otc.build_frequency_table([])
    assert table.total == 0
    fx = _build(trace_id="freq-untrained", n_candidates=5)
    ranking = otc.FrequencyBaselineV1(table).rank(fx.trace)
    assert ranking is not None
    assert ranking.application_order == tuple(
        sorted(a.application_id for a in fx.trace.legal_set.operator_actions)
    )
    assert all(abs(p - 0.2) < 1e-9 for p in ranking.probabilities)


def test_current_scorer_baseline_ranks_by_declared_scores() -> None:
    fx = _build(
        trace_id="scorer-full",
        n_candidates=5,
        current_scores_by_index={0: 0.1, 1: 0.2, 2: 0.9, 3: 0.3, 4: 0.05},
    )
    ranking = otc.CurrentScorerBaselineV1().rank(fx.trace)
    assert ranking is not None
    best_id = fx.library.dry_run(
        fx.pack, fx.state0, OPERATOR_ID, (BoundArgumentV1("value", fx.ref_by_index[2]),), fx.provenance
    ).application_id
    assert ranking.application_order[0] == best_id
    assert ranking.reliability is otc.RankingReliability.EXACT
    assert ranking.approximate is False


def test_current_scorer_baseline_returns_none_when_unscored() -> None:
    fx = _build(trace_id="scorer-none", n_candidates=5)
    assert otc.CurrentScorerBaselineV1().rank(fx.trace) is None


def test_current_scorer_baseline_partial_scores_are_marked_approximate() -> None:
    fx = _build(
        trace_id="scorer-partial",
        n_candidates=5,
        current_scores_by_index={0: 0.9, 1: 0.1},
    )
    ranking = otc.CurrentScorerBaselineV1().rank(fx.trace)
    assert ranking is not None
    assert len(ranking.application_order) == 5
    assert ranking.reliability is otc.RankingReliability.APPROXIMATE
    assert ranking.approximate is True
    assert "partial" in ranking.reliability_reason


def test_descriptor_similarity_baseline_ties_on_structurally_identical_candidates() -> None:
    # A single operator with a single required VALUE slot produces
    # structurally-identical candidates (same operator_id, same arity, same
    # proof checks) -- descriptor similarity is honestly uninformative here,
    # and the baseline must tie deterministically rather than fabricate signal.
    fx = _build(trace_id="descsim-tie", n_candidates=5)
    ranking = otc.DescriptorSimilarityBaselineV1().rank(fx.trace)
    assert ranking is not None
    assert ranking.application_order == tuple(
        sorted(a.application_id for a in fx.trace.legal_set.operator_actions)
    )
    probs = ranking.probabilities
    assert probs is not None
    assert max(probs) - min(probs) < 1e-9


def test_oracle_accepted_set_comparator_ranks_accepted_first() -> None:
    fx = _build(trace_id="oracle-1", n_candidates=5, accepted_indices=(1, 3))
    ranking = otc.OracleAcceptedSetComparatorV1().rank(fx.trace)
    assert ranking is not None
    accepted_ids = set(fx.trace.accepted_application_ids)
    assert set(ranking.application_order[:2]) == accepted_ids
    assert ranking.probabilities is not None
    assert all(
        abs(p - 0.5) < 1e-9
        for aid, p in zip(ranking.application_order, ranking.probabilities)
        if aid in accepted_ids
    )
    assert otc.reciprocal_rank(ranking, frozenset(accepted_ids)) == 1.0


def test_oracle_accepted_set_comparator_no_accepted_actions() -> None:
    fx = _build(trace_id="oracle-none", n_candidates=5)
    ranking = otc.OracleAcceptedSetComparatorV1().rank(fx.trace)
    assert ranking is not None
    assert ranking.probabilities is None
    assert ranking.reliability is otc.RankingReliability.APPROXIMATE
    assert "no_accepted_actions_to_oracle" in ranking.reliability_reason


# --------------------------------------------------------------------------- #
# Synthetic teacher
# --------------------------------------------------------------------------- #


def test_synthetic_teacher_never_reads_query_or_evaluator_only_fields() -> None:
    """A teacher genuinely blind to presentation metadata and evaluator-only
    labels must rank identically no matter how the query is perturbed."""
    fx = _build(
        trace_id="teacher-blind",
        n_candidates=5,
        accepted_indices=(2,),
        current_scores_by_index={0: 0.9, 1: 0.05, 2: 0.02, 3: 0.02, 4: 0.01},
    )
    teacher = otc.SyntheticDescriptorTeacherV1()
    canonical = teacher.rank(otc.default_teacher_query(fx.trace))
    perturbed_query = otc.perturb_candidate_order(
        otc.perturb_opaque_id_labels(
            otc.perturb_description_length(
                otc.perturb_prompt_template(otc.default_teacher_query(fx.trace), seed=1), seed=2
            ),
            seed=3,
        ),
        seed=4,
    )
    perturbed = teacher.rank(perturbed_query)
    assert canonical is not None and perturbed is not None
    assert canonical.application_order == perturbed.application_order
    assert canonical.probabilities == perturbed.probabilities


def test_synthetic_teacher_reports_exact_reliability_for_complete_coverage() -> None:
    fx = _build(trace_id="teacher-exact", n_candidates=5)
    ranking = otc.SyntheticDescriptorTeacherV1().rank(otc.default_teacher_query(fx.trace))
    assert ranking is not None
    assert ranking.reliability is otc.RankingReliability.EXACT
    assert "exact_finite_set_logits" in ranking.reliability_reason
    assert abs(sum(ranking.probabilities) - 1.0) < 1e-9


def test_synthetic_teacher_empty_legal_set() -> None:
    fx = _build(trace_id="teacher-empty", n_candidates=0)
    ranking = otc.SyntheticDescriptorTeacherV1().rank(otc.default_teacher_query(fx.trace))
    assert ranking is not None
    assert ranking.application_order == ()


# --------------------------------------------------------------------------- #
# PromptedTeacherV1 (SLM-429 / VAR3-01): a real, prompted-LLM teacher
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, output_parsed) -> None:
        self.output_parsed = output_parsed


class _FakeResponses:
    def __init__(self, output_parsed, *, raises: Exception | None = None) -> None:
        self._output_parsed = output_parsed
        self._raises = raises
        self.calls: list[dict] = []

    def parse(self, *, model, input, text_format):  # noqa: A002 - matches openai SDK signature
        self.calls.append({"model": model, "input": input, "text_format": text_format})
        if self._raises is not None:
            raise self._raises
        return _FakeResponse(self._output_parsed)


class _FakeClient:
    def __init__(self, output_parsed=None, *, raises: Exception | None = None) -> None:
        self.responses = _FakeResponses(output_parsed, raises=raises)


def test_prompted_teacher_prompt_never_leaks_scores_or_accepted_or_real_ids() -> None:
    fx = _build(
        trace_id="prompted-anti-leak",
        n_candidates=5,
        accepted_indices=(2,),
        current_scores_by_index={0: 0.91234, 1: 0.05, 2: 0.02, 3: 0.019, 4: 0.001},
    )
    query = otc.default_teacher_query(fx.trace)
    prompt = pt.build_prompt(query)

    assert fx.trace.current_scores is not None
    for score in fx.trace.current_scores.values():
        assert str(score) not in prompt
    for accepted_id in fx.trace.accepted_application_ids:
        assert accepted_id not in prompt
    for action in fx.trace.legal_set.operator_actions:
        assert action.application_id not in prompt
        # Only the caller-controlled presentation label may appear, never the
        # real application_id -- assert the rendered label is present instead.
        assert query.opaque_id_labels[action.application_id] in prompt


def test_prompted_teacher_prompt_changes_under_perturbation() -> None:
    # Unlike SyntheticDescriptorTeacherV1, a real prompted teacher's *prompt
    # text* legitimately varies with presentation -- this is what
    # run_perturbation_robustness measures, not something this module hides.
    fx = _build(trace_id="prompted-perturb", n_candidates=4)
    canonical = otc.default_teacher_query(fx.trace)
    perturbed = otc.perturb_candidate_order(canonical, seed=1)
    assert pt.build_prompt(canonical) != pt.build_prompt(perturbed)


def test_prompted_teacher_maps_a_well_formed_response_to_a_valid_ranking() -> None:
    fx = _build(trace_id="prompted-happy", n_candidates=3)
    query = otc.default_teacher_query(fx.trace)
    labels = [query.opaque_id_labels[aid] for aid in reversed(query.candidate_order)]
    client = _FakeClient(
        pt.PromptedTeacherResponseV1(ranked_candidate_labels=labels, confidence=[0.6, 0.3, 0.1])
    )
    teacher = pt.PromptedTeacherV1(client=client)

    ranking = teacher.rank(query)

    assert ranking is not None
    assert ranking.application_order == tuple(reversed(query.candidate_order))
    assert ranking.probabilities == (0.6, 0.3, 0.1)
    assert len(client.responses.calls) == 1


def test_prompted_teacher_returns_none_for_unrecognized_label() -> None:
    fx = _build(trace_id="prompted-bad-label", n_candidates=3)
    query = otc.default_teacher_query(fx.trace)
    client = _FakeClient(
        pt.PromptedTeacherResponseV1(ranked_candidate_labels=["not-a-real-label"])
    )
    teacher = pt.PromptedTeacherV1(client=client)
    assert teacher.rank(query) is None


def test_prompted_teacher_returns_none_for_incomplete_ranking() -> None:
    fx = _build(trace_id="prompted-partial", n_candidates=3)
    query = otc.default_teacher_query(fx.trace)
    only_one_label = [query.opaque_id_labels[query.candidate_order[0]]]
    client = _FakeClient(pt.PromptedTeacherResponseV1(ranked_candidate_labels=only_one_label))
    teacher = pt.PromptedTeacherV1(client=client)
    assert teacher.rank(query) is None


def test_prompted_teacher_returns_none_on_client_failure() -> None:
    fx = _build(trace_id="prompted-error", n_candidates=3)
    query = otc.default_teacher_query(fx.trace)
    client = _FakeClient(raises=RuntimeError("simulated transport failure"))
    teacher = pt.PromptedTeacherV1(client=client)
    assert teacher.rank(query) is None


def test_prompted_teacher_empty_legal_set_never_calls_the_client() -> None:
    fx = _build(trace_id="prompted-empty", n_candidates=0)
    query = otc.default_teacher_query(fx.trace)
    client = _FakeClient(pt.PromptedTeacherResponseV1(ranked_candidate_labels=["unused"]))
    teacher = pt.PromptedTeacherV1(client=client)

    ranking = teacher.rank(query)

    assert ranking is not None
    assert ranking.application_order == ()
    assert client.responses.calls == []


def test_prompted_teacher_requires_openai_or_an_injected_client() -> None:
    with pytest.raises(RuntimeError, match="install slm-training\\[research\\]"):
        pt.PromptedTeacherV1()


def test_prompted_teacher_satisfies_the_operator_teacher_adapter_protocol() -> None:
    # OperatorTeacherAdapter is a structural (non-runtime-checkable) Protocol
    # -- assert the shape directly rather than isinstance() against it.
    teacher = pt.PromptedTeacherV1(client=_FakeClient(None))
    assert isinstance(teacher.source_id, str)
    assert callable(teacher.rank)


# --------------------------------------------------------------------------- #
# Approximate coverage -> explicit reliability metadata on every comparator
# --------------------------------------------------------------------------- #


def test_partial_coverage_trace_marks_every_comparator_approximate() -> None:
    fx = _build(
        trace_id="approx-1",
        n_candidates=5,
        max_combinations_per_operator=1,
        approximate=True,
    )
    assert fx.trace.coverage.value == "partial"
    comparators = [
        otc.CompilerOrderBaselineV1(),
        otc.DescriptorSimilarityBaselineV1(),
        otc.FrequencyBaselineV1(otc.build_frequency_table([])),
        otc.OracleAcceptedSetComparatorV1(),
    ]
    for comparator in comparators:
        ranking = comparator.rank(fx.trace)
        assert ranking is not None
        assert ranking.approximate is True, comparator.source_id
        assert ranking.reliability is otc.RankingReliability.APPROXIMATE, comparator.source_id

    teacher_ranking = otc.SyntheticDescriptorTeacherV1().rank(otc.default_teacher_query(fx.trace))
    assert teacher_ranking is not None
    assert teacher_ranking.approximate is True
    assert teacher_ranking.reliability is otc.RankingReliability.APPROXIMATE
    assert "approximate_preference" in teacher_ranking.reliability_reason


# --------------------------------------------------------------------------- #
# Metrics: edge cases (empty legal set, single candidate, ties)
# --------------------------------------------------------------------------- #


def _ranking(order: tuple[str, ...], probs: tuple[float, ...] | None = None) -> otc.RankingV1:
    return otc.RankingV1(
        trace_id="t",
        source_id="s",
        application_order=order,
        probabilities=probs,
        reliability=otc.RankingReliability.EXACT,
        reliability_reason="test",
        approximate=False,
    )


def test_reciprocal_rank_edge_cases() -> None:
    assert otc.reciprocal_rank(_ranking(()), frozenset()) is None
    assert otc.reciprocal_rank(_ranking(("a",)), frozenset({"a"})) == 1.0
    assert otc.reciprocal_rank(_ranking(("a", "b", "c")), frozenset({"c"})) == pytest.approx(1 / 3)
    assert otc.reciprocal_rank(_ranking(("a", "b")), frozenset({"z"})) == 0.0


def test_top_k_recall_edge_cases() -> None:
    assert otc.top_k_recall(_ranking(()), frozenset(), 3) is None
    assert otc.top_k_recall(_ranking(("a",)), frozenset({"a"}), 3) == 1.0
    assert otc.top_k_recall(_ranking(("a", "b", "c")), frozenset({"a", "z"}), 1) == pytest.approx(0.5)


def test_ndcg_at_k_edge_cases() -> None:
    assert otc.ndcg_at_k(_ranking(()), frozenset(), 3) is None
    assert otc.ndcg_at_k(_ranking(("a",)), frozenset({"a"}), 3) == 1.0
    perfect = _ranking(("a", "b", "c"))
    assert otc.ndcg_at_k(perfect, frozenset({"a"}), 3) == 1.0
    worst = _ranking(("c", "b", "a"))
    assert otc.ndcg_at_k(worst, frozenset({"a"}), 3) < 1.0


def test_accepted_set_mass_requires_probabilities() -> None:
    assert otc.accepted_set_mass(_ranking(("a", "b")), frozenset({"a"})) is None
    ranking = _ranking(("a", "b"), (0.7, 0.3))
    assert otc.accepted_set_mass(ranking, frozenset({"a"})) == pytest.approx(0.7)
    assert otc.accepted_set_mass(ranking, frozenset()) == 0.0


def test_expected_calibration_error_edge_cases() -> None:
    assert otc.expected_calibration_error([]) is None
    perfectly_calibrated = [(1.0, True)] * 10 + [(0.0, False)] * 10
    assert otc.expected_calibration_error(perfectly_calibrated) == pytest.approx(0.0)
    overconfident = [(1.0, False)] * 5
    assert otc.expected_calibration_error(overconfident) == pytest.approx(1.0)


def test_selective_risk_aurc_edge_cases() -> None:
    assert otc.selective_risk_aurc([]) is None
    assert otc.selective_risk_aurc([(0.9, True), (0.1, True)]) == pytest.approx(0.0)
    assert otc.selective_risk_aurc([(0.9, False), (0.1, False)]) == pytest.approx(1.0)


def test_pearson_and_spearman_correlation_edge_cases() -> None:
    assert otc.pearson_correlation([1.0], [1.0]) is None  # n < 2
    assert otc.pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # zero variance
    assert otc.pearson_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert otc.pearson_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert otc.spearman_correlation([1.0, 2.0, 3.0], [1.0, 5.0, 100.0]) == pytest.approx(1.0)
    assert otc.spearman_correlation([1.0], [1.0]) is None


# --------------------------------------------------------------------------- #
# Perturbation robustness
# --------------------------------------------------------------------------- #


def test_ranking_distance_identical_and_reversed() -> None:
    a = _ranking(("a", "b", "c"))
    assert otc.ranking_distance(a, a) == 0.0
    reversed_ranking = _ranking(("c", "b", "a"))
    assert otc.ranking_distance(a, reversed_ranking) == pytest.approx(1.0)


def test_ranking_distance_rejects_mismatched_candidate_sets() -> None:
    with pytest.raises(ValueError, match="identical candidate set"):
        otc.ranking_distance(_ranking(("a", "b")), _ranking(("a", "c")))


def test_run_perturbation_robustness_is_zero_for_the_synthetic_teacher() -> None:
    traces = [
        _build(trace_id=f"pert-{i}", n_candidates=5, accepted_indices=(i % 5,)).trace
        for i in range(5)
    ]
    report = otc.run_perturbation_robustness(
        otc.SyntheticDescriptorTeacherV1(), traces, bound=otc.DECLARED_PERTURBATION_BOUND
    )
    assert report.within_bound is True
    assert all(distance == 0.0 for distance in report.kind_distances.values())
    assert all(n == len(traces) for n in report.n_traces_compared.values())


def test_run_perturbation_robustness_skips_single_candidate_traces() -> None:
    traces = [_build(trace_id="pert-single", n_candidates=1).trace]
    report = otc.run_perturbation_robustness(otc.SyntheticDescriptorTeacherV1(), traces)
    assert report.within_bound is True
    assert all(n == 0 for n in report.n_traces_compared.values())


# --------------------------------------------------------------------------- #
# Stop rule: a genuine passing scenario and a genuine failing scenario
# --------------------------------------------------------------------------- #


def _paired(baseline: str, ci_low: float, n_paired: int = 10) -> otc.PairedComparisonV1:
    return otc.PairedComparisonV1(
        metric_name=otc.PRIMARY_METRIC,
        family=None,
        baseline_source_id=baseline,
        teacher_source_id="teacher:fake",
        n_paired=n_paired,
        mean_diff=abs(ci_low) + 0.05,
        ci_low=ci_low,
        ci_high=abs(ci_low) + 0.2,
        win_rate=0.9 if ci_low > 0 else 0.4,
    )


def _perturbation(within_bound: bool) -> otc.PerturbationRobustnessReportV1:
    return otc.PerturbationRobustnessReportV1(
        kind_distances={"candidate_order": 0.0 if within_bound else 0.5},
        bound=0.0,
        within_bound=within_bound,
        n_traces_compared={"candidate_order": 10},
    )


def test_evaluate_stop_rule_passing_scenario() -> None:
    comparisons = [_paired(baseline_id, ci_low=0.05) for baseline_id in otc.REQUIRED_BASELINE_IDS]
    verdict = otc.evaluate_stop_rule(
        paired_comparisons=comparisons,
        perturbation_report=_perturbation(within_bound=True),
        verifier_regret_correlation=0.4,
    )
    assert verdict.go is True
    assert verdict.reasons == ()
    assert verdict.beats_required_baselines is True
    assert verdict.failing_baselines == ()
    assert verdict.correlation_positive is True


def test_evaluate_stop_rule_failing_scenario_combines_every_reason() -> None:
    comparisons = [
        _paired(otc.REQUIRED_BASELINE_IDS[0], ci_low=-0.1),  # teacher loses
        *[_paired(bid, ci_low=0.05) for bid in otc.REQUIRED_BASELINE_IDS[1:]],
    ]
    verdict = otc.evaluate_stop_rule(
        paired_comparisons=comparisons,
        perturbation_report=_perturbation(within_bound=False),
        verifier_regret_correlation=-0.2,
    )
    assert verdict.go is False
    assert verdict.beats_required_baselines is False
    assert verdict.failing_baselines == (otc.REQUIRED_BASELINE_IDS[0],)
    assert verdict.perturbation_within_bound is False
    assert verdict.correlation_positive is False
    assert len(verdict.reasons) == 3


def test_evaluate_stop_rule_missing_comparison_is_insufficient_data_not_a_win() -> None:
    comparisons = [_paired(bid, ci_low=0.05) for bid in otc.REQUIRED_BASELINE_IDS[1:]]
    verdict = otc.evaluate_stop_rule(
        paired_comparisons=comparisons,
        perturbation_report=_perturbation(within_bound=True),
        verifier_regret_correlation=0.4,
    )
    assert verdict.go is False
    assert verdict.insufficient_data_baselines == (otc.REQUIRED_BASELINE_IDS[0],)
    assert otc.REQUIRED_BASELINE_IDS[0] in verdict.failing_baselines


def test_evaluate_stop_rule_zero_n_paired_is_insufficient_data() -> None:
    comparisons = [
        _paired(otc.REQUIRED_BASELINE_IDS[0], ci_low=0.0, n_paired=0),
        *[_paired(bid, ci_low=0.05) for bid in otc.REQUIRED_BASELINE_IDS[1:]],
    ]
    verdict = otc.evaluate_stop_rule(
        paired_comparisons=comparisons,
        perturbation_report=_perturbation(within_bound=True),
        verifier_regret_correlation=0.4,
    )
    assert verdict.go is False
    assert otc.REQUIRED_BASELINE_IDS[0] in verdict.insufficient_data_baselines


# --------------------------------------------------------------------------- #
# Frequency table training/eval leakage guard
# --------------------------------------------------------------------------- #


def test_build_frequency_table_ignores_traces_with_no_accepted_actions() -> None:
    fx = _build(trace_id="freq-no-accept", n_candidates=5)
    table = otc.build_frequency_table([fx.trace])
    assert table.total == 0


# --------------------------------------------------------------------------- #
# End-to-end orchestration
# --------------------------------------------------------------------------- #


def test_compare_operator_teacher_ceiling_rejects_empty_eval_traces() -> None:
    with pytest.raises(ValueError, match="at least one eval trace"):
        otc.compare_operator_teacher_ceiling(
            eval_traces=(),
            frequency_training_traces=(),
            teacher=otc.SyntheticDescriptorTeacherV1(),
        )


def test_compare_operator_teacher_ceiling_rejects_train_eval_leakage() -> None:
    fx = _build(trace_id="leak-1", n_candidates=5, accepted_indices=(0,))
    with pytest.raises(ValueError, match="leak into eval_traces"):
        otc.compare_operator_teacher_ceiling(
            eval_traces=[fx.trace],
            frequency_training_traces=[fx.trace],
            teacher=otc.SyntheticDescriptorTeacherV1(),
        )


def test_compare_operator_teacher_ceiling_rejects_duplicate_eval_trace_ids() -> None:
    fx = _build(trace_id="dup-1", n_candidates=5, accepted_indices=(0,))
    with pytest.raises(ValueError, match="unique trace_id"):
        otc.compare_operator_teacher_ceiling(
            eval_traces=[fx.trace, fx.trace],
            frequency_training_traces=[],
            teacher=otc.SyntheticDescriptorTeacherV1(),
        )


def test_compare_operator_teacher_ceiling_end_to_end_report_shape(tmp_path) -> None:
    training = [
        _build(trace_id=f"e2e-train-{i}", n_candidates=5, accepted_indices=(2,)).trace
        for i in range(6)
    ]
    eval_traces = [
        _build(
            trace_id=f"e2e-eval-{i}",
            n_candidates=5,
            accepted_indices=(2,),
            current_scores_by_index={0: 0.1, 1: 0.1, 2: 0.6, 3: 0.1, 4: 0.1},
            supervision=SupervisionSource.GOLD if i % 2 == 0 else SupervisionSource.ON_POLICY,
        ).trace
        for i in range(6)
    ]

    report = otc.compare_operator_teacher_ceiling(
        eval_traces=eval_traces,
        frequency_training_traces=training,
        teacher=otc.SyntheticDescriptorTeacherV1(),
        n_bootstrap=200,
    )

    assert report.schema == "dsh4-02.v1"
    assert report.n_eval_traces == 6
    assert report.n_frequency_training_traces == 6
    assert set(report.metrics_overall) == {
        "baseline:frequency",
        "baseline:compiler_order",
        "baseline:current_scorer",
        "baseline:descriptor_similarity",
        "oracle:accepted_set",
        "teacher:synthetic_descriptor_v1",
    }
    assert report.metrics_overall["oracle:accepted_set"]["mrr"]["mean"] == pytest.approx(1.0)
    assert report.perturbation.within_bound is True
    assert isinstance(report.stop_rule, otc.StopRuleVerdictV1)
    assert "wiring" in " ".join(report.caveats)

    payload = report.to_dict()
    serialized = json.dumps(payload)  # must be JSON-serializable
    assert '"schema": "dsh4-02.v1"' in serialized or "dsh4-02.v1" in serialized

    out_path = tmp_path / "report.json"
    otc.write_operator_teacher_ceiling_report(out_path, report)
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert reloaded["stop_rule"]["primary_metric"] == "mrr"
    assert reloaded["n_eval_traces"] == 6
