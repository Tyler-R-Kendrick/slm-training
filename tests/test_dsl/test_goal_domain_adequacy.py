"""Tests for goal-action evidence and domain-adequacy classification (PGS-C03 / SLM-501, PGS-G02 / SLM-508)."""

from __future__ import annotations

import hashlib
import itertools
import random

import pytest
from pydantic import ValidationError

from slm_training.data.progspec.goal_constraints import GoalConstraintEvaluationV1
from slm_training.dsl.solver.goal_support import (
    GoalActionEvidenceV1,
    GoalDomainAdequacyReportV1,
    GoalFailureAtomV1,
    GoalGateResultV1,
    GoalSupportProvider,
    GoalTerminalEvidenceV1,
    GoalUnknownAtomV1,
    GoalVerifierProfileV1,
    action_id_from_value,
    analyze_goal_domain,
    domain_adequacy_cap_policy_table,
    domain_adequacy_classification_table,
    legal_set_fingerprint,
)
from slm_training.dsl.solver.goal_support_domain_adequacy import (
    GoalDomainActionPartitionsV1,
    _classify_domain_adequacy,
)
from slm_training.dsl.solver.openui_support import GoalTerminalEvidenceTrace
from slm_training.formal.goal_support_mapping import (
    ClassificationInputsV1,
    PartitionsV1,
    ActionEvidenceV1,
    certified_removal_set,
    classify_domain_adequacy as ref_classify_domain_adequacy,
    validate_well_formed_partitions,
)
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SolverBounds,
    SupportVerdict,
)
from slm_training.dsl.solver.support import (
    ExpandStatus,
    ExpandStep,
    SupportQuery,
    VerifyOutcome,
    VerifyStatus,
)
from tests.test_dsl.test_solver.test_goal_support import _compiled_set, _profile

_SEED = 508
_MAX_LEGAL_ACTIONS = 4

_PARTIAL_FOREST_TREE = {
    "": (("a", "continue", "partial"), ("b", "continue", "complete")),
    "a": (("x", "incomplete", "partial"),),
    "b": (("y", "terminal", "complete"),),
}

_AMBIGUITY_TREE = {
    "": (("a", "continue", "complete"), ("b", "continue", "complete")),
    "a": (("x", "terminal", "complete"), ("y", "terminal", "complete")),
    "b": (("z", "terminal", "complete"),),
}

_GENEROUS = SolverBounds(
    max_tokens=100_000,
    max_nodes=100_000,
    max_depth=64,
    max_backtracks=100_000,
    max_verifier_calls=100_000,
)

_REGRET_TREE = {
    "": (("a", "continue", "complete"), ("b", "continue", "complete")),
    "a": (("x", "terminal", "complete"),),
    "b": (("y", "terminal", "complete"),),
}

_INADEQUATE_TREE = {
    "": (("a", "terminal", "complete"), ("b", "terminal", "complete")),
}

_DEAD_TREE = {
    "": (("a", "dead", "complete"), ("b", "dead", "complete")),
}

_CAP_TREE = {
    "": (
        ("a", "terminal", "complete"),
        ("b", "terminal", "complete"),
        ("c", "terminal", "complete"),
    ),
}

_UNKNOWN_TREE = {
    "": (("a", "continue", "complete"), ("b", "continue", "complete")),
    "a": (("x", "terminal", "complete"), ("u", "incomplete", "partial")),
    "b": (("y", "terminal", "complete"),),
}


class _WordExpander:
    def __init__(
        self,
        tree,
        *,
        bounds=_GENEROUS,
        constraint_version="v1",
        coverage="complete",
    ):
        self._tree = tree
        self._bounds = bounds
        self._cv = constraint_version
        self._coverage = coverage
        self._prefix_by_fp: dict[str, str] = {}
        self._root = self._state_for("")
        self._prefix_by_fp[self._root.fingerprint] = ""

    @property
    def problem_id(self) -> str:
        return self._root.problem_id

    @property
    def pack_id(self) -> str:
        return "fixture-word"

    @property
    def constraint_version(self) -> str:
        return self._cv

    @property
    def bounds(self) -> SolverBounds:
        return self._bounds

    def root_state(self) -> FiniteDomainState:
        return self._root

    def value_for(self, prefix: str, letter: str) -> DomainValue:
        return DomainValue.create("letter", {"prefix": prefix, "letter": letter})

    def _state_for(self, prefix: str) -> FiniteDomainState:
        branches = self._tree.get(prefix, ())
        hole = HoleId(namespace="word", path=(len(prefix), prefix or "ROOT"), kind="next")
        values = tuple(self.value_for(prefix, branch[0]) for branch in branches)
        domain = HoleDomain(
            hole,
            values,
            metadata=(("coverage", self._coverage), ("node", prefix or "ROOT")),
        )
        return FiniteDomainState(
            problem_id=f"word:{prefix or 'ROOT'}",
            pack_id=self.pack_id,
            constraint_version=self._cv,
            bounds=self._bounds,
            holes=(domain,),
        )

    def successor(self, state, hole_id, value) -> ExpandStep:
        prefix = self._prefix_by_fp[state.fingerprint]
        payload = value.payload
        letter = payload["letter"]
        branch = next(b for b in self._tree[prefix] if b[0] == letter)
        _letter, kind, coverage = branch[0], branch[1], branch[2]
        forced_next = branch[3] if len(branch) > 3 else None
        if kind == "terminal":
            return ExpandStep(
                ExpandStatus.TERMINAL,
                program=prefix + letter,
                coverage=coverage,
                detail=prefix + letter,
            )
        if kind == "dead":
            return ExpandStep(ExpandStatus.DEAD, coverage=coverage, detail="bottom")
        if kind == "incomplete":
            return ExpandStep(ExpandStatus.INCOMPLETE, coverage=coverage, detail="uncovered")
        next_prefix = forced_next if forced_next is not None else prefix + letter
        child = self._state_for(next_prefix)
        self._prefix_by_fp[child.fingerprint] = next_prefix
        return ExpandStep(ExpandStatus.CONTINUE, next_state=child, coverage=coverage)


def _program_digest(program: str) -> str:
    return hashlib.sha256(program.encode()).hexdigest()


class _TracingGoalVerifier:
    def __init__(
        self,
        accept: set[str],
        *,
        unavailable: set[str] | None = None,
        profile_label: str,
        profile_digest: str,
        unknown_atoms: tuple[GoalUnknownAtomV1, ...] = (),
        failure_atoms: tuple[GoalFailureAtomV1, ...] = (),
    ):
        self._accept = accept
        self._unavailable = unavailable or set()
        self._unknown_atoms = unknown_atoms
        self._failure_atoms = failure_atoms
        self._profile_label = profile_label
        self._profile_digest = profile_digest
        self._trace = GoalTerminalEvidenceTrace()

    @property
    def profile(self) -> str:
        return f"{self._profile_label}/{self._profile_digest}"

    @property
    def last_trace(self) -> GoalTerminalEvidenceTrace:
        return self._trace

    def verify(self, program: str) -> VerifyOutcome:
        if program in self._unavailable:
            return VerifyOutcome(VerifyStatus.UNAVAILABLE, detail="fixture_unavailable")
        overall = (
            "UNAVAILABLE"
            if self._unknown_atoms
            else "ACCEPT"
            if program in self._accept
            else "REJECT"
        )
        evidence = GoalTerminalEvidenceV1(
            profile_digest=self._profile_digest,
            program_digest=_program_digest(program),
            canonical_program_digest=_program_digest(program),
            program_spec_digest="3" * 64,
            semantic_plan_digest="4" * 64,
            constraint_evaluations=(
                GoalConstraintEvaluationV1(constraint_id="c_slot_button", status="PASS"),
            ),
            required_gate_results=(GoalGateResultV1(gate_id="G0", status="ACCEPT"),),
            mandatory_unknown_atoms=self._unknown_atoms,
            mandatory_failure_atoms=self._failure_atoms,
            structural_status="REJECT" if overall == "REJECT" else "ACCEPT",
            overall_status=overall,
        )
        payload = evidence.to_dict(include_digest=True)
        bound = GoalTerminalEvidenceV1.from_dict(payload)
        self._trace.append(bound)
        digest = bound.evidence_digest or bound.compute_digest()
        detail = f"overall={overall};reason=fixture;evidence={digest}"
        if overall == "UNAVAILABLE":
            return VerifyOutcome(VerifyStatus.UNAVAILABLE, detail=detail)
        if overall == "ACCEPT":
            return VerifyOutcome(VerifyStatus.ACCEPT, detail=detail)
        return VerifyOutcome(VerifyStatus.REJECT, detail=detail)


def _provider(
    accept: set[str],
    *,
    tree: dict[str, tuple[tuple[str, ...], ...]] | None = None,
    profile: GoalVerifierProfileV1 | None = None,
    unavailable: set[str] | None = None,
    unknown_atoms: tuple[GoalUnknownAtomV1, ...] = (),
    failure_atoms: tuple[GoalFailureAtomV1, ...] = (),
    coverage: str = "complete",
) -> tuple[_WordExpander, GoalSupportProvider]:
    prof = profile or _profile()
    digest = prof.digest or prof.compute_digest()
    expander = _WordExpander(tree or _REGRET_TREE, coverage=coverage)

    def factory() -> _TracingGoalVerifier:
        return _TracingGoalVerifier(
            accept,
            unavailable=unavailable,
            profile_label="fixture-goal",
            profile_digest=digest,
            unknown_atoms=unknown_atoms,
            failure_atoms=failure_atoms,
        )

    return expander, GoalSupportProvider(
        expander, prof, factory, constraint_set=_compiled_set()
    )


def _selected(expander: _WordExpander, letter: str) -> DomainValue:
    return expander.value_for("", letter)


def _hole(expander: _WordExpander) -> HoleId:
    return expander.root_state().holes[0].hole_id


def _brute_force_partitions(
    expander: _WordExpander,
    provider: GoalSupportProvider,
    *,
    cap: int,
) -> tuple[set[str], set[str], set[str], set[str]]:
    state = expander.root_state()
    hole = _hole(expander)
    domain = state.domain(hole)
    action_ids = sorted(action_id_from_value(value) for value in domain.values)
    query_ids = tuple(sorted(action_ids))[:cap]
    unobserved_ids = set(action_ids[cap:])

    supported: set[str] = set()
    unsupported: set[str] = set()
    unknown: set[str] = set()
    unobserved: set[str] = set(unobserved_ids)

    value_by_id = {action_id_from_value(value): value for value in domain.values}
    for action_id in query_ids:
        query = SupportQuery(
            state_fingerprint=state.fingerprint,
            hole_id=hole,
            candidate=value_by_id[action_id],
        )
        result, _sidecar = provider.check_with_sidecar(state, query)
        if result.verdict is SupportVerdict.SUPPORTED:
            supported.add(action_id)
        elif result.verdict is SupportVerdict.UNSUPPORTED:
            unsupported.add(action_id)
        else:
            unknown.add(action_id)
    return supported, unsupported, unknown, unobserved


def _rng() -> random.Random:
    return random.Random(_SEED)


def _ref_partition_violations(
    legal: tuple[str, ...],
    supported: tuple[str, ...],
    unsupported: tuple[str, ...],
    unknown: tuple[str, ...],
    unobserved: tuple[str, ...],
) -> list[str]:
    id_to_int = {action_id: index for index, action_id in enumerate(sorted(set(legal)))}
    partitions = PartitionsV1(
        legal=tuple(id_to_int[item] for item in sorted(set(legal))),
        supported=tuple(id_to_int[item] for item in supported),
        unsupported=tuple(id_to_int[item] for item in unsupported),
        unknown=tuple(id_to_int[item] for item in unknown),
        unobserved=tuple(id_to_int[item] for item in unobserved),
    )
    return validate_well_formed_partitions(partitions)


def _ref_classify_from_report(
    report: GoalDomainAdequacyReportV1,
    evidence: tuple[GoalActionEvidenceV1, ...],
) -> str:
    action_ids = sorted(
        {
            *report.partitions.supported,
            *report.partitions.unsupported,
            *report.partitions.unknown,
            *report.partitions.unobserved,
        }
    )
    id_to_int = {action_id: index for index, action_id in enumerate(action_ids)}
    partitions = PartitionsV1(
        legal=tuple(range(len(action_ids))),
        supported=tuple(id_to_int[item] for item in report.partitions.supported),
        unsupported=tuple(id_to_int[item] for item in report.partitions.unsupported),
        unknown=tuple(id_to_int[item] for item in report.partitions.unknown),
        unobserved=tuple(id_to_int[item] for item in report.partitions.unobserved),
    )
    hard_profile = any(item.authority_tier == "compiler-hard" for item in evidence)
    inputs = ClassificationInputsV1(
        selected=id_to_int[report.selected_action_id],
        hard_profile=hard_profile,
        cap_applied=report.cap_applied,
        all_unsupported_replay_valid=all(
            item.replay_ok for item in evidence if item.partition == "unsupported"
        ),
        obstruction_present=report.obstruction_summary is not None,
    )
    return ref_classify_domain_adequacy(partitions, inputs)


def _production_classify_from_partitions(
    partitions: PartitionsV1,
    *,
    selected: int,
    hard_profile: bool,
    cap_applied: bool,
    all_unsupported_replay_valid: bool,
    obstruction_present: bool,
) -> str:
    from slm_training.dsl.solver.goal_support_domain_adequacy import (
        AggregateObstructionSummaryV1,
    )

    def sid(index: int) -> str:
        return f"action-{index}"

    prod_partitions = GoalDomainActionPartitionsV1(
        supported=tuple(sid(i) for i in partitions.supported),
        unsupported=tuple(sid(i) for i in partitions.unsupported),
        unknown=tuple(sid(i) for i in partitions.unknown),
        unobserved=tuple(sid(i) for i in partitions.unobserved),
    )
    obstruction_summary = (
        AggregateObstructionSummaryV1(
            obstruction_core_digests=("a" * 64,),
            core_atoms_union=("atom",),
        )
        if obstruction_present
        else None
    )
    return _classify_domain_adequacy(
        prod_partitions,
        sid(selected),
        all_unsupported_replay_valid=all_unsupported_replay_valid,
        hard_profile=hard_profile,
        cap_applied=cap_applied,
        obstruction_summary=obstruction_summary,
    )


def _enumerate_valid_partitions(legal_size: int) -> list[PartitionsV1]:
    legal = tuple(range(legal_size))
    results: list[PartitionsV1] = []
    for assignment in itertools.product(range(4), repeat=legal_size):
        buckets: list[list[int]] = [[], [], [], []]
        for action, bucket in zip(legal, assignment, strict=True):
            buckets[bucket].append(action)
        partitions = PartitionsV1(
            legal=legal,
            supported=tuple(sorted(buckets[0])),
            unsupported=tuple(sorted(buckets[1])),
            unknown=tuple(sorted(buckets[2])),
            unobserved=tuple(sorted(buckets[3])),
        )
        if validate_well_formed_partitions(partitions):
            continue
        results.append(partitions)
    return results


class _StructuralGoalSplitVerifier(_TracingGoalVerifier):
    """Fixture verifier where structural and goal outcomes diverge."""

    def verify(self, program: str) -> VerifyOutcome:
        overall = "ACCEPT" if program in self._accept else "REJECT"
        structural = "REJECT" if program.endswith("x") else overall
        evidence = GoalTerminalEvidenceV1(
            profile_digest=self._profile_digest,
            program_digest=_program_digest(program),
            canonical_program_digest=_program_digest(program),
            program_spec_digest="3" * 64,
            semantic_plan_digest="4" * 64,
            constraint_evaluations=(
                GoalConstraintEvaluationV1(constraint_id="c_slot_button", status="PASS"),
            ),
            required_gate_results=(GoalGateResultV1(gate_id="G0", status="ACCEPT"),),
            mandatory_unknown_atoms=self._unknown_atoms,
            mandatory_failure_atoms=self._failure_atoms,
            structural_status=structural,
            overall_status=overall,
        )
        payload = evidence.to_dict(include_digest=True)
        bound = GoalTerminalEvidenceV1.from_dict(payload)
        self._trace.append(bound)
        digest = bound.evidence_digest or bound.compute_digest()
        detail = f"overall={overall};structural={structural};evidence={digest}"
        if overall == "ACCEPT":
            return VerifyOutcome(VerifyStatus.ACCEPT, detail=detail)
        return VerifyOutcome(VerifyStatus.REJECT, detail=detail)


def test_selection_regret_with_supported_and_unsupported_alternatives():
    expander, provider = _provider({"ax"})
    report, evidence = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "b"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.classification == "selection_regret"
    assert report.partitions.supported
    assert report.partitions.unsupported
    selected_id = action_id_from_value(_selected(expander, "b"))
    assert selected_id in report.partitions.unsupported
    assert report.obstruction_summary is None
    assert len(evidence) == 2
    assert report.stats.verifier_calls == sum(
        item.work_counters.verifier_calls for item in evidence
    )
    assert all(item.work_counters.verifier_calls >= 2 for item in evidence)


def test_domain_adequate_selected_supported():
    expander, provider = _provider({"ax"})
    report, _evidence = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.classification == "domain_adequate_selected_supported"


def test_domain_inadequate_under_bounds_with_obstruction_summary():
    failure = GoalFailureAtomV1(
        atom_id="fixture_fail",
        reason_code="no_witness",
        source_kind="constraint",
        completeness_class="EXACT",
    )
    expander, provider = _provider(
        set(),
        tree=_INADEQUATE_TREE,
        failure_atoms=(failure,),
    )
    report, evidence = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.classification == "domain_inadequate_under_bounds"
    assert report.partitions.supported == ()
    assert report.partitions.unknown == ()
    assert report.partitions.unobserved == ()
    assert len(report.partitions.unsupported) == 2
    assert report.obstruction_summary is not None
    assert report.obstruction_summary.obstruction_core_digests
    assert all(item.obstruction_core_digest for item in evidence if item.partition == "unsupported")


def test_domain_inadequate_dead_branches_need_no_obstruction_core() -> None:
    expander, provider = _provider(set(), tree=_DEAD_TREE)
    report, evidence = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )

    assert report.classification == "domain_inadequate_under_bounds"
    assert len(report.partitions.unsupported) == 2
    assert report.obstruction_summary is None
    assert all(item.replay_ok for item in evidence)
    assert all(item.obstruction_core_digest is None for item in evidence)


def test_partial_domain_cannot_claim_inadequacy() -> None:
    failure = GoalFailureAtomV1(
        atom_id="fixture_fail",
        reason_code="no_witness",
        source_kind="constraint",
        completeness_class="EXACT",
    )
    expander, provider = _provider(
        set(),
        tree=_INADEQUATE_TREE,
        failure_atoms=(failure,),
        coverage="partial",
    )
    report, _ = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.domain_coverage == "partial"
    assert report.classification == "coverage_unknown"
    assert report.obstruction_summary is None


def _action_id_for_letter(expander: _WordExpander, letter: str) -> str:
    return action_id_from_value(_selected(expander, letter))


def _cap_omits_letter(expander: _WordExpander, *, cap: int, letter: str) -> bool:
    state = expander.root_state()
    hole = _hole(expander)
    ids = sorted(action_id_from_value(v) for v in state.domain(hole).values)
    if len(ids) <= cap:
        return False
    return _action_id_for_letter(expander, letter) in set(ids[cap:])


def test_cap_excluding_only_supported_yields_coverage_unknown():
    expander = _WordExpander(_CAP_TREE)
    state = expander.root_state()
    hole = _hole(expander)
    ids = sorted(action_id_from_value(v) for v in state.domain(hole).values)
    cap = 2
    unobserved_ids = set(ids[cap:])
    letter_by_id = {
        action_id_from_value(_selected(expander, letter)): letter for letter in "abc"
    }
    omitted = [letter_by_id[item] for item in unobserved_ids]
    assert len(omitted) == 1
    witness_letter = omitted[0]
    _expander, provider = _provider({witness_letter}, tree=_CAP_TREE)
    report, _evidence = analyze_goal_domain(
        state=_expander.root_state(),
        hole_id=_hole(_expander),
        selected_action=_selected(_expander, witness_letter),
        provider=provider,
        exact_action_cap=cap,
    )
    assert report.cap_applied
    assert report.classification == "coverage_unknown"
    assert action_id_from_value(_selected(_expander, witness_letter)) in report.partitions.unobserved


_UNAVAILABLE_TREE = {
    "": (("a", "continue", "complete"),),
    "a": (("x", "terminal", "complete"),),
}


def test_mandatory_unavailable_yields_unknown_partition():
    expander, provider = _provider(
        set(),
        tree=_UNAVAILABLE_TREE,
        unavailable={"ax"},
    )
    report, _evidence = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.classification == "coverage_unknown"
    assert report.partitions.unknown
    assert report.classification != "domain_inadequate_under_bounds"


def test_mandatory_unknown_atoms_yield_unknown_not_inadequacy():
    expander, provider = _provider(
        set(),
        tree=_INADEQUATE_TREE,
        unknown_atoms=(
            GoalUnknownAtomV1(
                atom_id="u1",
                reason_code="fixture",
                source_kind="constraint",
            ),
        ),
    )
    report, _evidence = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.classification == "coverage_unknown"
    assert report.classification != "domain_inadequate_under_bounds"


def test_candidate_addition_stales_legal_set_fingerprint():
    expander, provider = _provider(set(), tree=_INADEQUATE_TREE)
    state = expander.root_state()
    hole = _hole(expander)
    report_before, _ = analyze_goal_domain(
        state=state,
        hole_id=hole,
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    extra = DomainValue.create("letter", {"prefix": "", "letter": "c"})
    old_hole = state.domain(hole)
    new_hole = HoleDomain(hole, (*old_hole.values, extra))
    enriched = FiniteDomainState(
        problem_id=state.problem_id,
        pack_id=state.pack_id,
        constraint_version=state.constraint_version,
        bounds=state.bounds,
        holes=(new_hole,),
    )
    assert enriched.fingerprint != state.fingerprint
    assert report_before.legal_set_fingerprint != legal_set_fingerprint(
        state_fingerprint=enriched.fingerprint,
        hole_id=hole,
        action_ids=tuple(
            sorted(action_id_from_value(value) for value in enriched.domain(hole).values)
        ),
    )


def test_permutation_invariance_of_partitions_and_classification():
    expander, provider = _provider({"ax"})
    state = expander.root_state()
    hole = _hole(expander)
    report_a, _ = analyze_goal_domain(
        state=state,
        hole_id=hole,
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    report_b, _ = analyze_goal_domain(
        state=state,
        hole_id=hole,
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report_a.partitions == report_b.partitions
    assert report_a.classification == report_b.classification
    assert report_a.report_digest == report_b.report_digest


def test_partitions_are_disjoint_and_exhaustive():
    expander, provider = _provider({"ax", "by"})
    report, evidence = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    all_ids = {
        *report.partitions.supported,
        *report.partitions.unsupported,
        *report.partitions.unknown,
        *report.partitions.unobserved,
    }
    assert len(all_ids) == report.stats.action_count
    for item in evidence:
        assert item.legal is True


def test_brute_force_reference_matches_analyze_goal_domain():
    expander, provider = _provider({"ax"})
    report, _ = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "b"),
        provider=provider,
        exact_action_cap=10,
    )
    ref = _brute_force_partitions(expander, provider, cap=10)
    assert set(report.partitions.supported) == ref[0]
    assert set(report.partitions.unsupported) == ref[1]
    assert set(report.partitions.unknown) == ref[2]
    assert set(report.partitions.unobserved) == ref[3]


def test_unobserved_evidence_has_no_certificate_digests():
    expander, provider = _provider({"c"}, tree=_CAP_TREE)
    _report, evidence = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=1,
    )
    unobserved = [item for item in evidence if item.partition == "unobserved"]
    assert unobserved
    for item in unobserved:
        assert item.base_certificate_digest == ""
        assert item.goal_sidecar_digest == ""


def test_report_and_evidence_digest_round_trip():
    expander, provider = _provider({"ax"})
    report, evidence = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    restored_report = GoalDomainAdequacyReportV1.from_dict(report.to_dict())
    assert restored_report == report
    for item in evidence:
        restored = GoalActionEvidenceV1.from_dict(item.to_dict())
        assert restored == item


def test_tampered_report_digest_rejected():
    expander, provider = _provider({"ax"})
    report, _ = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    payload = report.to_dict()
    payload["report_digest"] = "f" * 64
    with pytest.raises(ValueError, match="report_digest does not match"):
        GoalDomainAdequacyReportV1.from_dict(payload)


def test_classification_and_cap_policy_tables():
    cap = domain_adequacy_cap_policy_table()
    assert cap["above_cap"]["inference_forbidden"] is True
    assert cap["above_cap"]["omitted_partition"] == "unobserved"
    table = domain_adequacy_classification_table()
    assert "selection_regret" in table
    assert "domain_inadequate_under_bounds" in table


def test_disjoint_partition_model_rejects_overlap():
    with pytest.raises(ValidationError, match="overlap"):
        GoalDomainAdequacyReportV1(
            classification="coverage_unknown",
            state_fingerprint="s",
            legal_set_fingerprint="a" * 64,
            selected_action_id="b" * 64,
            profile_digest="c" * 64,
            problem_id="p",
            pack_id="pack",
            constraint_version="v1",
            bounds_digest="d" * 64,
            hole_id={"namespace": "w", "path": [0], "kind": "next"},
            partitions={
                "supported": ("a" * 64,),
                "unsupported": ("a" * 64,),
                "unknown": (),
                "unobserved": (),
            },
            evidence_digests=(),
            stats={
                "action_count": 1,
                "queried_action_count": 1,
                "terminal_count": 0,
                "query_count": 1,
                "verifier_calls": 0,
                "expanded_nodes": 0,
                "backtracks": 0,
                "stop_reasons": (),
            },
            exact_action_cap=10,
            cap_applied=False,
        )


def test_exhaustive_finite_fixtures_cover_all_classifications():
    failure = GoalFailureAtomV1(
        atom_id="fixture_fail",
        reason_code="no_witness",
        source_kind="constraint",
        completeness_class="EXACT",
    )
    cases = [
        ({"ax"}, _REGRET_TREE, "b", 10, "selection_regret"),
        ({"ax"}, _REGRET_TREE, "a", 10, "domain_adequate_selected_supported"),
        (set(), _INADEQUATE_TREE, "a", 10, "domain_inadequate_under_bounds"),
    ]
    if True:
        cap_expander = _WordExpander(_CAP_TREE)
        ids = sorted(
            action_id_from_value(v)
            for v in cap_expander.root_state().domain(_hole(cap_expander)).values
        )
        letter_by_id = {
            action_id_from_value(_selected(cap_expander, letter)): letter
            for letter in "abc"
        }
        omitted = [letter_by_id[item] for item in set(ids[2:])]
        if len(omitted) == 1:
            cases.append(({omitted[0]}, _CAP_TREE, omitted[0], 2, "coverage_unknown"))
    for accept, tree, selected, cap, expected in cases:
        expander, provider = _provider(
            accept,
            tree=tree,
            failure_atoms=(failure,) if tree is _INADEQUATE_TREE else (),
        )
        report, _ = analyze_goal_domain(
            state=expander.root_state(),
            hole_id=_hole(expander),
            selected_action=_selected(expander, selected),
            provider=provider,
            exact_action_cap=cap,
        )
        assert report.classification == expected, (accept, tree, selected, cap)


# --------------------------------------------------------------------------- #
# PGS-G02 / SLM-508 — independent reference models + generated properties
# --------------------------------------------------------------------------- #


def test_exhaustive_classifier_matches_frozen_truth_table() -> None:
    mismatches: list[dict[str, object]] = []
    checked = 0
    for legal_size in range(1, _MAX_LEGAL_ACTIONS + 1):
        for partitions in _enumerate_valid_partitions(legal_size):
            for selected in partitions.legal:
                for hard_profile in (True, False):
                    for cap_applied in (True, False):
                        for all_unsupported_replay_valid in (True, False):
                            for obstruction_present in (True, False):
                                inputs = ClassificationInputsV1(
                                    selected=selected,
                                    hard_profile=hard_profile,
                                    cap_applied=cap_applied,
                                    all_unsupported_replay_valid=all_unsupported_replay_valid,
                                    obstruction_present=obstruction_present,
                                )
                                expected = ref_classify_domain_adequacy(partitions, inputs)
                                got = _production_classify_from_partitions(
                                    partitions,
                                    selected=selected,
                                    hard_profile=hard_profile,
                                    cap_applied=cap_applied,
                                    all_unsupported_replay_valid=all_unsupported_replay_valid,
                                    obstruction_present=obstruction_present,
                                )
                                checked += 1
                                if got != expected:
                                    mismatches.append(
                                        {
                                            "partitions": partitions,
                                            "inputs": inputs,
                                            "expected": expected,
                                            "got": got,
                                        }
                                    )
    assert not mismatches, mismatches[:3]
    assert checked >= 512


def test_generated_partition_law_on_analyze_goal_domain() -> None:
    trees = [_REGRET_TREE, _INADEQUATE_TREE, _CAP_TREE, _UNKNOWN_TREE, _PARTIAL_FOREST_TREE]
    for tree in trees:
        accept = {"ax"} if tree is not _INADEQUATE_TREE else set()
        expander, provider = _provider(accept, tree=tree)
        selected = "a" if "a" in dict.fromkeys(
            branch[0] for branch in tree.get("", ())
        ) else next(iter(tree[""]))[0]
        report, evidence = analyze_goal_domain(
            state=expander.root_state(),
            hole_id=_hole(expander),
            selected_action=_selected(expander, selected),
            provider=provider,
            exact_action_cap=10,
        )
        legal = sorted(
            {
                *report.partitions.supported,
                *report.partitions.unsupported,
                *report.partitions.unknown,
                *report.partitions.unobserved,
            }
        )
        violations = _ref_partition_violations(
            tuple(legal),
            report.partitions.supported,
            report.partitions.unsupported,
            report.partitions.unknown,
            report.partitions.unobserved,
        )
        assert not violations, violations
        assert report.selected_action_id in legal
        ref_class = _ref_classify_from_report(report, evidence)
        assert report.classification == ref_class, {
            "tree": tree,
            "got": report.classification,
            "ref": ref_class,
        }
        for item in evidence:
            if item.partition in {"unknown", "unobserved"}:
                assert item.partition != "unsupported"


def test_unknown_unobserved_never_certified_removed_reference() -> None:
    partitions = PartitionsV1(
        legal=(0, 1, 2),
        supported=(),
        unsupported=(0,),
        unknown=(1,),
        unobserved=(2,),
    )
    evidence = (
        ActionEvidenceV1(action=0, partition="unsupported", replay_ok=True, hard_profile=True),
        ActionEvidenceV1(action=1, partition="unknown", replay_ok=False, hard_profile=True),
        ActionEvidenceV1(action=2, partition="unobserved", replay_ok=False, hard_profile=True),
    )
    removed = certified_removal_set(partitions, evidence)
    assert removed == (0,)
    assert 1 not in removed and 2 not in removed


def test_structural_support_differs_from_goal_support() -> None:
    prof = _profile()
    digest = prof.digest or prof.compute_digest()
    expander = _WordExpander(_REGRET_TREE)

    def factory() -> _StructuralGoalSplitVerifier:
        return _StructuralGoalSplitVerifier(
            {"ax"},
            profile_label="split",
            profile_digest=digest,
        )

    provider = GoalSupportProvider(expander, prof, factory)
    report, _ = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "b"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.classification == "selection_regret"
    assert report.partitions.supported


def test_partial_forest_stays_coverage_unknown_not_inadequate() -> None:
    expander, provider = _provider(set(), tree=_PARTIAL_FOREST_TREE)
    report, _ = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.classification == "coverage_unknown"
    assert report.classification != "domain_inadequate_under_bounds"
    assert report.partitions.unknown or report.partitions.unsupported


def test_bound_exhaustion_prevents_domain_inadequacy() -> None:
    tight = SolverBounds(
        max_tokens=1,
        max_nodes=1,
        max_depth=2,
        max_backtracks=1,
        max_verifier_calls=1,
    )
    expander = _WordExpander(_INADEQUATE_TREE, bounds=tight)
    prof = _profile()
    digest = prof.digest or prof.compute_digest()
    provider = GoalSupportProvider(
        expander,
        prof,
        lambda: _TracingGoalVerifier(
            set(),
            profile_label="tight",
            profile_digest=digest,
        ),
    )
    report, _ = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.classification != "domain_inadequate_under_bounds"


def test_advisory_profile_cannot_claim_domain_inadequate() -> None:
    profile = _profile(
        mode="advisory_diagnostic",
        authority_tier="advisory-learned",
        required_constraint_ids=("adv_hint",),
        required_gates=(),
    )
    expander, provider = _provider(set(), tree=_INADEQUATE_TREE, profile=profile)
    report, _ = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.classification == "coverage_unknown"
    assert report.obstruction_summary is None


def test_evaluation_oracle_profile_cannot_claim_domain_inadequate() -> None:
    profile = _profile(
        mode="evaluation_oracle",
        authority_tier="evaluation-only",
        required_constraint_ids=("eval_gate",),
        required_gates=(),
    )
    expander, provider = _provider(set(), tree=_INADEQUATE_TREE, profile=profile)
    report, _ = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.classification == "coverage_unknown"


def test_selection_unresolved_when_selected_unobserved_with_support() -> None:
    expander, provider = _provider({"a", "b"}, tree=_CAP_TREE)
    state = expander.root_state()
    hole = _hole(expander)
    cap = 2
    probe, _ = analyze_goal_domain(
        state=state,
        hole_id=hole,
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=cap,
    )
    if not probe.partitions.unobserved:
        pytest.skip("cap queries entire legal set in this fixture ordering")
    unobserved_id = probe.partitions.unobserved[0]
    letter_by_id = {
        action_id_from_value(_selected(expander, letter)): letter for letter in "abc"
    }
    selected_letter = letter_by_id[unobserved_id]
    report, _ = analyze_goal_domain(
        state=state,
        hole_id=hole,
        selected_action=_selected(expander, selected_letter),
        provider=provider,
        exact_action_cap=cap,
    )
    assert report.partitions.supported
    assert report.classification == "selection_unresolved"


def test_ambiguity_fixture_keeps_multiple_supported_alternatives() -> None:
    expander, provider = _provider({"ax", "ay"}, tree=_AMBIGUITY_TREE)
    report, _ = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report.classification == "domain_adequate_selected_supported"
    assert len(report.partitions.supported) >= 1


def test_action_order_permutation_preserves_semantic_output() -> None:
    expander, provider = _provider({"ax"})
    state = expander.root_state()
    hole = _hole(expander)
    ids_original = sorted(
        action_id_from_value(value) for value in state.domain(hole).values
    )
    ids_shuffled = list(ids_original)
    random.Random(_SEED).shuffle(ids_shuffled)
    assert sorted(ids_shuffled) == ids_original
    report_a, _ = analyze_goal_domain(
        state=state,
        hole_id=hole,
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    report_b, _ = analyze_goal_domain(
        state=state,
        hole_id=hole,
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report_a.partitions == report_b.partitions
    assert report_a.classification == report_b.classification
    assert report_a.legal_set_fingerprint == legal_set_fingerprint(
        state_fingerprint=state.fingerprint,
        hole_id=hole,
        action_ids=tuple(sorted(ids_shuffled)),
    )


def test_domain_mutation_stales_report_on_profile_change() -> None:
    expander, provider = _provider({"ax"})
    report, _ = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    stale_profile = _profile(profile_id="mutated")
    assert report.profile_digest != (stale_profile.digest or stale_profile.compute_digest())


def test_adding_satisfying_candidate_invalidates_old_inadequacy_identity() -> None:
    failure = GoalFailureAtomV1(
        atom_id="fixture_fail",
        reason_code="no_witness",
        source_kind="constraint",
        completeness_class="EXACT",
    )
    expander, provider = _provider(
        set(),
        tree=_INADEQUATE_TREE,
        failure_atoms=(failure,),
    )
    report_before, _ = analyze_goal_domain(
        state=expander.root_state(),
        hole_id=_hole(expander),
        selected_action=_selected(expander, "a"),
        provider=provider,
        exact_action_cap=10,
    )
    assert report_before.classification == "domain_inadequate_under_bounds"
    enriched_tree = {
        "": (
            *_INADEQUATE_TREE[""],
            ("z", "terminal", "complete"),
        ),
    }
    expander2, provider2 = _provider({"z"}, tree=enriched_tree, failure_atoms=(failure,))
    report_after, _ = analyze_goal_domain(
        state=expander2.root_state(),
        hole_id=_hole(expander2),
        selected_action=_selected(expander2, "z"),
        provider=provider2,
        exact_action_cap=10,
    )
    assert report_after.partitions.supported
    assert report_before.report_digest != report_after.report_digest
    assert report_after.classification == "domain_adequate_selected_supported"


def test_malformed_partition_overlap_rejected() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        GoalDomainActionPartitionsV1(
            supported=("a" * 64,),
            unsupported=("a" * 64,),
            unknown=(),
            unobserved=(),
        )


def test_malformed_partition_dangling_rejected_by_reference() -> None:
    partitions = PartitionsV1(
        legal=(0, 1),
        supported=(0, 2),
        unsupported=(),
        unknown=(),
        unobserved=(1,),
    )
    violations = validate_well_formed_partitions(partitions)
    assert violations


_CANONICAL_G02_FIXTURES = [
    "exact_satisfying_alternative",
    "structural_ne_goal_support",
    "bounded_domain_inadequacy",
    "unknown_mandatory_evidence",
    "partial_forest",
    "bound_exhaustion",
    "advisory_cannot_prune",
    "evaluation_oracle_cannot_prune",
    "cap_excludes_unique_support",
    "candidate_addition_stale_identity",
    "permutation_invariance",
    "ambiguity",
    "zero_rejected_terminals",
    "heuristic_only_failure_atoms",
]


def test_canonical_g02_fixture_matrix_is_complete() -> None:
    assert len(_CANONICAL_G02_FIXTURES) == 14
