"""Tests for goal-aware support provider, sidecars, replay, and closure guard (PGS-C01)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from slm_training.data.progspec.goal_constraints import GoalConstraintEvaluationV1
from slm_training.dsl.solver.goal_support import (
    GoalGateResultV1,
    GoalSupportProvider,
    GoalSupportResultV1,
    GoalTerminalEvidenceV1,
    GoalVerifierProfileV1,
    exact_goal_closure,
    exact_goal_closure_authority_table,
    goal_support_backend_version,
    replay_goal_support_result,
)
from slm_training.dsl.solver.openui_support import GoalTerminalEvidenceTrace
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
    SupportCertificate,
    SupportQuery,
    VerifyOutcome,
    VerifyStatus,
)
from tests.test_dsl.test_solver.test_goal_support import _compiled_set, _profile

_GENEROUS = SolverBounds(
    max_tokens=100_000,
    max_nodes=100_000,
    max_depth=64,
    max_backtracks=100_000,
    max_verifier_calls=100_000,
)

_LINEAR_TREE = {
    "": (("a", "continue", "complete"), ("b", "continue", "complete")),
    "a": (("x", "terminal", "complete"), ("y", "terminal", "complete")),
    "b": (("z", "terminal", "complete"),),
}


class _WordExpander:
    def __init__(self, tree, *, bounds=_GENEROUS, constraint_version="v1"):
        self._tree = tree
        self._bounds = bounds
        self._cv = constraint_version
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
        domain = HoleDomain(hole, values, metadata=(("node", prefix or "ROOT"),))
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
    """Fixture verifier with query-local terminal-evidence trace."""

    def __init__(self, accept: set[str], *, profile_label: str, profile_digest: str):
        self._accept = accept
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
        overall = "ACCEPT" if program in self._accept else "REJECT"
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
            structural_status="ACCEPT" if overall == "ACCEPT" else "REJECT",
            overall_status=overall,
        )
        payload = evidence.to_dict(include_digest=True)
        bound = GoalTerminalEvidenceV1.from_dict(payload)
        self._trace.append(bound)
        digest = bound.evidence_digest or bound.compute_digest()
        detail = f"overall={overall};reason=fixture;evidence={digest}"
        if overall == "ACCEPT":
            return VerifyOutcome(VerifyStatus.ACCEPT, detail=detail)
        return VerifyOutcome(VerifyStatus.REJECT, detail=detail)


def _provider(
    accept: set[str],
    *,
    profile: GoalVerifierProfileV1 | None = None,
    tree=_LINEAR_TREE,
) -> tuple[_WordExpander, GoalSupportProvider]:
    prof = profile or _profile()
    digest = prof.digest or prof.compute_digest()
    expander = _WordExpander(tree)

    def factory() -> _TracingGoalVerifier:
        return _TracingGoalVerifier(accept, profile_label="fixture-goal", profile_digest=digest)

    return expander, GoalSupportProvider(
        expander, prof, factory, constraint_set=_compiled_set()
    )


def _query(expander: _WordExpander, letter: str) -> SupportQuery:
    root = expander.root_state()
    return SupportQuery(
        state_fingerprint=root.fingerprint,
        hole_id=root.holes[0].hole_id,
        candidate=expander.value_for("", letter),
    )


def test_supported_returns_base_certificate_and_sidecar():
    expander, provider = _provider({"ax"})
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    assert result.verdict is SupportVerdict.SUPPORTED
    assert sidecar.base_certificate_digest == result.certificate.digest
    assert sidecar.base_verdict == SupportVerdict.SUPPORTED.value
    assert sidecar.witness_terminal_evidence_digest is not None
    assert sidecar.witness_terminal_evidence_digest in sidecar.terminal_evidence_digests


def test_unsupported_and_unknown_sidecars_preserve_base_verdict():
    expander, provider = _provider({"ax"})
    unsupported, side_u = provider.check_with_sidecar(
        expander.root_state(), _query(expander, "b")
    )
    assert unsupported.verdict is SupportVerdict.UNSUPPORTED
    assert side_u.base_verdict == SupportVerdict.UNSUPPORTED.value
    assert side_u.witness_terminal_evidence_digest is None

    tree = {
        "": (("b", "continue", "complete"),),
        "b": (("z", "terminal", "complete"), ("w", "incomplete", "partial")),
    }
    expander2, provider2 = _provider(set(), tree=tree)
    unknown, side_k = provider2.check_with_sidecar(
        expander2.root_state(), _query(expander2, "b")
    )
    assert unknown.verdict is SupportVerdict.UNKNOWN
    assert side_k.base_verdict == SupportVerdict.UNKNOWN.value


def test_backend_version_is_profile_sensitive():
    profile_a = _profile(profile_id="a")
    profile_b = _profile(profile_id="b")
    assert goal_support_backend_version(profile_a) != goal_support_backend_version(profile_b)
    _, provider = _provider({"ax"}, profile=profile_a)
    assert provider.backend_version == goal_support_backend_version(profile_a)


def test_fresh_verifier_factory_prevents_trace_contamination():
    expander, provider = _provider({"ax"})
    _result_a, side_a = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    _result_b, side_b = provider.check_with_sidecar(expander.root_state(), _query(expander, "b"))
    assert side_a.terminal_evidence_digests != side_b.terminal_evidence_digests or (
        side_a.base_verdict != side_b.base_verdict
    )


def test_reused_mutable_verifier_factory_fails_closed():
    profile = _profile()
    expander = _WordExpander(_LINEAR_TREE)
    verifier = _TracingGoalVerifier(
        {"ax"},
        profile_label="fixture-goal",
        profile_digest=profile.digest,
    )
    provider = GoalSupportProvider(
        expander,
        profile,
        lambda: verifier,
        constraint_set=_compiled_set(),
    )
    with pytest.raises(ValueError, match="reused mutable query state"):
        provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))


def test_sidecar_digest_tampering_rejected():
    expander, provider = _provider({"ax"})
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    payload = sidecar.to_dict()
    payload["digest"] = "f" * 64
    with pytest.raises(ValueError, match="digest does not match"):
        GoalSupportResultV1.from_dict(payload)


def test_replay_goal_support_result_accepts_honest_sidecar():
    expander, provider = _provider({"ax"})
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    replay = replay_goal_support_result(
        result.certificate,
        sidecar,
        state=expander.root_state(),
        provider=provider,
    )
    assert replay.ok, replay.violations
    assert replay.verdict is SupportVerdict.SUPPORTED


def test_replay_rejects_stale_profile_digest():
    expander, provider = _provider({"ax"})
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    payload = sidecar.to_dict()
    payload["profile_digest"] = "f" * 64
    recomputed = GoalSupportResultV1(**{k: v for k, v in payload.items() if k != "digest"})
    payload["digest"] = recomputed.compute_digest()
    stale = GoalSupportResultV1.from_dict(payload)
    replay = replay_goal_support_result(
        result.certificate,
        stale,
        state=expander.root_state(),
        provider=provider,
    )
    assert not replay.ok
    assert any("profile_digest" in item for item in replay.violations)


def test_provider_replay_reports_stale_state_instead_of_raising():
    expander, provider = _provider({"ax"})
    state = expander.root_state()
    result, _sidecar = provider.check_with_sidecar(state, _query(expander, "a"))

    replay = provider.replay(
        result.certificate,
        state=replace(state, constraint_version="stale-constraint-set"),
    )

    assert not replay.ok
    assert replay.verdict is SupportVerdict.SUPPORTED
    assert replay.violations == ("goal support state identity does not match expander",)


def test_replay_failure_never_upgrades_unknown_to_unsupported():
    tree = {
        "": (("b", "continue", "complete"),),
        "b": (("z", "terminal", "complete"), ("w", "incomplete", "partial")),
    }
    expander, provider = _provider(set(), tree=tree)
    result, sidecar = provider.check_with_sidecar(
        expander.root_state(), _query(expander, "b")
    )
    assert result.verdict is SupportVerdict.UNKNOWN
    tampered = SupportCertificate.from_dict(
        json.loads(json.dumps(result.certificate.to_dict()))
    )
    tampered = replace(tampered, witness_digest="f" * 64)
    replay = replay_goal_support_result(
        tampered,
        sidecar,
        state=expander.root_state(),
        provider=provider,
    )
    assert not replay.ok
    assert replay.verdict is SupportVerdict.UNKNOWN


def test_exact_goal_closure_accepts_production_exact_hard_profile():
    expander, provider = _provider({"ax", "ay"})
    state = expander.root_state()
    result = exact_goal_closure(state, provider, max_queries=8)
    assert result.reached_fixed_point or result.stop_reason is not None
    assert result.counters.verifier_calls > result.counters.support_queries


def test_exact_goal_closure_rejects_evaluation_oracle_profile():
    profile = _profile(
        mode="evaluation_oracle",
        authority_tier="evaluation-only",
        required_constraint_ids=("eval_gate",),
        required_gates=(),
    )
    expander, provider = _provider({"ax"}, profile=profile)
    with pytest.raises(ValueError, match="rejects non-production"):
        exact_goal_closure(expander.root_state(), provider)


def test_exact_goal_closure_rejects_advisory_profile():
    profile = _profile(
        mode="advisory_diagnostic",
        authority_tier="advisory-learned",
        required_constraint_ids=("adv_hint",),
        required_gates=(),
    )
    expander, provider = _provider({"ax"}, profile=profile)
    with pytest.raises(ValueError, match="rejects non-production"):
        exact_goal_closure(expander.root_state(), provider)


def test_exact_goal_closure_rejects_verifier_hard_profile():
    profile = _profile(authority_tier="verifier-hard")
    expander, provider = _provider({"ax"}, profile=profile)
    with pytest.raises(ValueError, match="requires compiler-hard authority"):
        exact_goal_closure(expander.root_state(), provider)


def test_exact_goal_closure_authority_table():
    table = exact_goal_closure_authority_table()
    assert table["accepted"]["mode"] == "production_exact"
    assert table["accepted"]["authority_tiers"] == ["compiler-hard"]
    assert "evaluation_oracle" in table["rejected_modes"]
    assert "verifier-hard" in table["rejected_authority_tiers"]
    assert "advisory-learned" in table["rejected_authority_tiers"]


def test_sidecar_round_trip_json():
    expander, provider = _provider({"ax"})
    _result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    restored = GoalSupportResultV1.from_dict(json.loads(json.dumps(sidecar.to_dict())))
    assert restored == sidecar


def test_goal_support_provider_does_not_embed_vss_search():
    import inspect

    source = inspect.getsource(GoalSupportProvider)
    assert "while stack" not in source
    assert "ExpandStatus" not in source
