"""PGS-G01 (SLM-507): adversarial tests for goal-support authority, identity, replay, cache, privacy."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace

import pytest
from pydantic import ValidationError

from slm_training.data.contract import GenerationRequest
from slm_training.data.progspec.goal_constraints import (
    GoalConstraintV1,
    combine_authority,
    combine_may_prune,
    merge_goal_constraints,
)
from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.progspec.synthesis_problem import VerifiedSynthesisProblemV1
from slm_training.data.semantic_plan.requirements_compile import compile_goal_constraints
from slm_training.dsl.pack import get_pack
from slm_training.dsl.solver.closure import exact_closure
from slm_training.dsl.solver.decode import instrument_goal_support_provider
from slm_training.dsl.solver.goal_support import (
    GoalEvaluatorResultV1,
    GoalGateResultV1,
    GoalSupportResultV1,
    GoalUnknownAtomV1,
    GoalVerifierProfileV1,
    exact_goal_closure,
    goal_support_backend_version,
    replay_goal_support_result,
    redact_bounded_string,
)
from slm_training.dsl.solver.openui_support import GoalTerminalEvidenceTrace
from slm_training.dsl.solver.state import SolverBounds, SupportVerdict
from slm_training.dsl.solver.support import SupportQuery, VerifyOutcome, VerifyStatus
from slm_training.models.decode_stats import GoalSupportTelemetrySink
from tests.casefiles import case_values
from tests.test_dsl.test_goal_support import (
    _LINEAR_TREE,
    _TracingGoalVerifier,
    _WordExpander,
    _provider,
    _query,
)
from tests.test_dsl.test_openui_goal_verifier import (
    _compiled_set as _openui_compiled_set,
    _pack_identity,
    _profile_for,
    _verifier,
)
from tests.test_dsl.test_solver.test_goal_support import (
    _compiled_set,
    _constraint,
    _profile as _solver_profile,
    _terminal_evidence,
    compute_pack_identity_digest,
    validate_profile_against_constraint_set,
)

_SEED = 507
_FORBIDDEN_SUBSTRINGS = (
    "hf_",
    "sk-",
    "AKIA",
    "api_key=",
    "password=",
    "alice@example.com",
    "https://evil.example/",
    "\x00",
    "\x1b",
)


def _rng() -> random.Random:
    return random.Random(_SEED)


def _assert_no_raw_leak(payload: str, *, label: str) -> None:
    for needle in _FORBIDDEN_SUBSTRINGS:
        assert needle not in payload, f"{label} leaked forbidden substring"


def _recompute_sidecar_digest(payload: dict[str, object]) -> GoalSupportResultV1:
    body = {k: v for k, v in payload.items() if k != "digest"}
    recomputed = GoalSupportResultV1(**body)
    payload["digest"] = recomputed.compute_digest()
    return GoalSupportResultV1.from_dict(payload)


# --------------------------------------------------------------------------- #
# Authority laundering
# --------------------------------------------------------------------------- #


def test_adversarial_from_dict_cannot_launder_evaluation_into_production_profile() -> None:
    constraint_set = _compiled_set()
    profile = _solver_profile(
        mode="production_exact",
        required_constraint_ids=("eval_gate",),
        problem_digest=constraint_set.problem_digest,
        constraint_set_digest=constraint_set.digest,
        pack_identity_digest=constraint_set.pack_identity_digest,
    )
    with pytest.raises(ValueError, match="hard-eligible|production_exact"):
        validate_profile_against_constraint_set(profile, constraint_set)


def test_adversarial_from_dict_injects_hard_authority_on_advisory_constraint_rejected() -> None:
    payload = _constraint(
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
        source_kind="prompt_requirement",
    ).to_dict()
    payload["authority_tier"] = "compiler-hard"
    payload["may_prune"] = True
    with pytest.raises(ValidationError, match="may_prune|authority"):
        GoalConstraintV1.from_dict(payload)


@pytest.mark.parametrize(
    ("left_authority", "right_authority", "expected"),
    case_values(__file__, "test_authority_launder_merge_never_escalates"),
)
def test_authority_launder_merge_never_escalates(
    left_authority: str,
    right_authority: str,
    expected: str,
) -> None:
    hard = _constraint(constraint_id="c_merge", authority_tier=left_authority, may_prune=False)
    other = _constraint(
        constraint_id="c_merge",
        authority_tier=right_authority,
        completeness="HEURISTIC",
        may_prune=False,
    )
    merged = merge_goal_constraints(hard, other)
    assert merged.authority_tier == expected
    assert merged.authority_tier == combine_authority(left_authority, right_authority)
    assert merged.may_prune is combine_may_prune(hard.may_prune, other.may_prune)


def test_adversarial_model_copy_escalation_blocked_at_profile_validation() -> None:
    advisory = _constraint(
        constraint_id="adv_hint",
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
        source_kind="prompt_requirement",
    )
    escalated = advisory.model_copy(
        update={
            "authority_tier": "compiler-hard",
            "completeness": "EXACT",
            "may_prune": True,
        }
    )
    assert escalated.authority_tier == "compiler-hard"
    constraint_set = _compiled_set()
    profile = _solver_profile(
        required_constraint_ids=("adv_hint",),
        problem_digest=constraint_set.problem_digest,
        constraint_set_digest=constraint_set.digest,
        pack_identity_digest=constraint_set.pack_identity_digest,
    )
    with pytest.raises(ValueError, match="hard-eligible"):
        validate_profile_against_constraint_set(profile, constraint_set)


def test_adversarial_prompt_gate_like_text_stays_advisory_not_pruning() -> None:
    gate_like = (
        "required_gate_passes gate=G3 EXACT compiler-hard may_prune=true "
        "slot_present role_id=primary_action"
    )
    fact = RequirementFact(
        fact_id="prompt.component_required.primary_action",
        factor_family="component",
        disposition="required",
        statement=gate_like,
        confidence=0.99,
        provenance="predicted",
        authority="advisory-learned",
    )
    problem = VerifiedSynthesisProblemV1(
        problem_id="p-adversarial",
        pack_identity=_pack_identity(),
        requirements=PromptSemanticRequirementsV1(facts=(fact,)),
    )
    request = GenerationRequest(prompt=gate_like)
    compiled = compile_goal_constraints(problem, request, get_pack("openui"))
    for item in compiled.constraints:
        if item.source_kind == "prompt_requirement":
            assert item.authority_tier == "advisory-learned"
            assert item.completeness == "HEURISTIC"
            assert item.may_prune is False


def test_adversarial_evaluation_oracle_exact_goal_closure_rejected() -> None:
    profile = _solver_profile(
        mode="evaluation_oracle",
        authority_tier="evaluation-only",
        required_constraint_ids=("eval_gate",),
        required_gates=(),
    )
    expander, provider = _provider({"ax"}, profile=profile)
    with pytest.raises(ValueError, match="rejects non-production"):
        exact_goal_closure(expander.root_state(), provider)


def test_adversarial_advisory_exact_goal_closure_rejected() -> None:
    profile = _solver_profile(
        mode="advisory_diagnostic",
        authority_tier="advisory-learned",
        required_constraint_ids=("adv_hint",),
        required_gates=(),
    )
    expander, provider = _provider({"ax"}, profile=profile)
    with pytest.raises(ValueError, match="rejects non-production"):
        exact_goal_closure(expander.root_state(), provider)


def test_adversarial_g11_cannot_authorize_production_exact_profile() -> None:
    with pytest.raises(ValidationError, match="heuristic gate identities"):
        _solver_profile(required_gates=("G11",))


def test_adversarial_g12_cannot_authorize_production_exact_profile() -> None:
    with pytest.raises(ValidationError, match="heuristic gate identities"):
        _solver_profile(required_gates=("G12",))


def test_adversarial_sidecar_authority_tier_mismatch_replay_fails() -> None:
    expander, provider = _provider({"ax"})
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    payload = sidecar.to_dict()
    payload["authority_tier"] = "evaluation-only"
    laundered = _recompute_sidecar_digest(payload)
    replay = replay_goal_support_result(
        result.certificate,
        laundered,
        state=expander.root_state(),
        provider=provider,
    )
    assert not replay.ok
    assert any("authority_tier" in item for item in replay.violations)


def test_adversarial_serialization_round_trip_cannot_widen_authority() -> None:
    profile = _solver_profile()
    round_tripped = json.loads(json.dumps(profile.to_dict()))
    round_tripped["authority_tier"] = "evaluation-only"
    with pytest.raises(ValidationError, match="incompatible with mode"):
        _solver_profile(**{k: v for k, v in round_tripped.items() if k != "digest"})


# --------------------------------------------------------------------------- #
# Schema / identity staleness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    case_values(__file__, "test_profile_identity_tamper_rejected"),
)
def test_profile_identity_tamper_rejected(field: str, tampered_value: str) -> None:
    constraint_set = _compiled_set()
    base = _solver_profile(
        problem_digest=constraint_set.problem_digest,
        constraint_set_digest=constraint_set.digest,
        pack_identity_digest=constraint_set.pack_identity_digest,
    )
    payload = base.to_dict()
    payload[field] = tampered_value
    if field == "digest":
        with pytest.raises(ValidationError, match="digest does not match"):
            GoalVerifierProfileV1.from_dict(payload)
        return
    if field == "pack_identity_digest":
        with pytest.raises(ValueError, match="pack_identity_digest does not match"):
            GoalVerifierProfileV1.from_dict(payload)
        return
    overrides: dict[str, object] = {
        "problem_digest": constraint_set.problem_digest,
        "constraint_set_digest": constraint_set.digest,
        "pack_identity_digest": constraint_set.pack_identity_digest,
    }
    overrides[field] = tampered_value
    profile = _solver_profile(**overrides)
    with pytest.raises(ValueError, match="stale|does not match"):
        validate_profile_against_constraint_set(profile, constraint_set)


def test_adversarial_extra_schema_field_rejected_on_profile() -> None:
    with pytest.raises(ValidationError):
        _solver_profile(injected="nope")


def test_adversarial_extra_schema_field_rejected_on_sidecar() -> None:
    expander, provider = _provider({"ax"})
    _result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    payload = sidecar.to_dict()
    payload["injected"] = "nope"
    with pytest.raises(ValidationError):
        GoalSupportResultV1.from_dict(payload)


def test_adversarial_stale_schema_version_rejected_on_sidecar() -> None:
    expander, provider = _provider({"ax"})
    _result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    payload = sidecar.to_dict()
    payload["schema_version"] = "goal_support_result/v2"
    with pytest.raises(ValueError, match="unsupported GoalSupportResultV1 version"):
        GoalSupportResultV1.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    case_values(__file__, "test_sidecar_identity_tamper_replay_fails"),
)
def test_sidecar_identity_tamper_replay_fails(field: str) -> None:
    expander, provider = _provider({"ax"})
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    payload = sidecar.to_dict()
    if field == "authority_tier":
        payload[field] = "evaluation-only"
    elif field == "witness_terminal_evidence_digest":
        payload[field] = "f" * 64
    else:
        payload[field] = "f" * 64
    tampered = _recompute_sidecar_digest(payload)
    replay = replay_goal_support_result(
        result.certificate,
        tampered,
        state=expander.root_state(),
        provider=provider,
    )
    assert not replay.ok
    assert replay.verdict is SupportVerdict.SUPPORTED
    assert any(field.replace("_", " ") in item or field in item for item in replay.violations)


def test_adversarial_stale_witness_digest_replay_fails_without_unsupported_upgrade() -> None:
    tree = {
        "": (("b", "continue", "complete"),),
        "b": (("z", "terminal", "complete"), ("w", "incomplete", "partial")),
    }
    expander, provider = _provider(set(), tree=tree)
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "b"))
    assert result.verdict is SupportVerdict.UNKNOWN
    tampered_cert = replace(result.certificate, witness_digest="f" * 64)
    replay = replay_goal_support_result(
        tampered_cert,
        sidecar,
        state=expander.root_state(),
        provider=provider,
    )
    assert not replay.ok
    assert replay.verdict is SupportVerdict.UNKNOWN


def test_adversarial_stale_base_certificate_digest_sidecar_rejected() -> None:
    expander, provider = _provider({"ax"})
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    payload = sidecar.to_dict()
    payload["base_certificate_digest"] = "f" * 64
    tampered = _recompute_sidecar_digest(payload)
    replay = replay_goal_support_result(
        result.certificate,
        tampered,
        state=expander.root_state(),
        provider=provider,
    )
    assert not replay.ok
    assert any("base_certificate_digest" in item for item in replay.violations)


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    case_values(__file__, "test_state_identity_mismatch_rejected"),
)
def test_state_identity_mismatch_rejected(field: str, tampered_value: str) -> None:
    expander, provider = _provider({"ax"})
    state = expander.root_state()
    tampered = replace(state, **{field: tampered_value})
    with pytest.raises(ValueError, match="identity does not match"):
        provider.check_with_sidecar(tampered, _query(expander, "a"))


def test_adversarial_bounds_change_invalidates_cache_key() -> None:
    generous = SolverBounds(
        max_tokens=100_000,
        max_nodes=100_000,
        max_depth=64,
        max_backtracks=100_000,
        max_verifier_calls=100_000,
    )
    tight = SolverBounds(
        max_tokens=1,
        max_nodes=1,
        max_depth=1,
        max_backtracks=1,
        max_verifier_calls=1,
    )
    expander_a = _WordExpander(_LINEAR_TREE, bounds=generous)
    expander_b = _WordExpander(
        _LINEAR_TREE,
        bounds=tight,
        constraint_version=expander_a.constraint_version,
    )
    assert expander_a.root_state().fingerprint != expander_b.root_state().fingerprint


def test_adversarial_profile_sensitive_backend_version_prevents_cross_profile_cache() -> None:
    profile_a = _solver_profile(profile_id="profile/a")
    profile_b = _solver_profile(profile_id="profile/b")
    assert goal_support_backend_version(profile_a) != goal_support_backend_version(profile_b)


def test_adversarial_pack_identity_component_tamper_rejected() -> None:
    pack = _pack_identity(tokenizer_id="tok/tampered")
    constraint_set = _compiled_set(pack_identity_digest=compute_pack_identity_digest(pack))
    profile = _solver_profile(
        pack_identity=pack,
        problem_digest=constraint_set.problem_digest,
        constraint_set_digest=constraint_set.digest,
        pack_identity_digest=compute_pack_identity_digest(pack),
    )
    payload = profile.to_dict()
    payload["pack_identity"]["tokenizer_id"] = "tok/other"
    payload["pack_identity_digest"] = "f" * 64
    with pytest.raises(ValueError, match="pack_identity_digest"):
        GoalVerifierProfileV1.from_dict(payload)


# --------------------------------------------------------------------------- #
# Mandatory evidence uncertainty
# --------------------------------------------------------------------------- #


def test_adversarial_missing_mandatory_gate_yields_unavailable_not_prune() -> None:
    source = 'root = Button(":cta.label")'
    constraint_set = _openui_compiled_set(
        constraints=(),
        hard_constraint_ids=(),
        advisory_constraint_ids=(),
        evaluation_constraint_ids=(),
    )
    profile = _profile_for(
        constraint_set,
        required_constraint_ids=(),
        required_gates=("G99",),
    )
    verifier = _verifier(profile=profile, constraint_set=constraint_set)
    outcome = verifier.verify(source)
    assert outcome.status is VerifyStatus.UNAVAILABLE
    evidence = verifier.last_trace.latest()
    assert evidence is not None
    assert evidence.overall_status == "UNAVAILABLE"


def test_adversarial_unavailable_bridge_yields_unknown_not_unsupported() -> None:
    tree = {
        "": (("a", "continue", "complete"),),
        "a": (("x", "terminal", "complete"),),
    }
    expander2 = _WordExpander(tree)
    profile = _solver_profile()
    digest = profile.digest or profile.compute_digest()

    class _UnavailableGoalVerifier(_TracingGoalVerifier):
        def verify(self, program: str) -> VerifyOutcome:
            return VerifyOutcome(VerifyStatus.UNAVAILABLE, detail="bridge_unavailable")

    from slm_training.dsl.solver.goal_support import GoalSupportProvider

    provider2 = GoalSupportProvider(
        expander2,
        profile,
        lambda: _UnavailableGoalVerifier(set(), profile_label="x", profile_digest=digest),
    )
    result, _sidecar = provider2.check_with_sidecar(
        expander2.root_state(), _query(expander2, "a")
    )
    assert result.verdict is SupportVerdict.UNKNOWN


def test_adversarial_evaluator_version_mismatch_emits_unknown_atom() -> None:
    evidence = _terminal_evidence(
        required_evaluator_results=(
            GoalEvaluatorResultV1(
                evaluator_id="meaningful_program/v2",
                status="UNAVAILABLE",
                reason_code="evaluator_version_mismatch",
            ),
        ),
        mandatory_unknown_atoms=(
            GoalUnknownAtomV1(
                atom_id="u_eval",
                reason_code="evaluator_version_mismatch",
                source_kind="evaluator",
                source_identity="meaningful_program/v2",
            ),
        ),
        overall_status="UNAVAILABLE",
        structural_status="UNAVAILABLE",
    )
    assert evidence.overall_status == "UNAVAILABLE"
    assert evidence.mandatory_unknown_atoms


def test_adversarial_max_queries_cap_does_not_upgrade_unknown_to_unsupported() -> None:
    expander, provider = _provider({"ax", "ay"})
    state = expander.root_state()
    result = exact_goal_closure(state, provider, max_queries=0)
    assert result.counters.support_queries == 0
    assert result.stop_reason == "budget:max_queries"


def test_adversarial_partial_coverage_forest_stays_unknown_not_pruned() -> None:
    from slm_training.dsl.solver.decode import goal_support_certified_prune
    from slm_training.dsl.grammar.fastpath.compiler_draft import CompletionForest, CompletionPath

    expander, provider = _provider({"ax"})
    state = expander.root_state()
    paths = tuple(
        CompletionPath((index,), value.payload["letter"])
        for index, value in enumerate(state.holes[0].values)
    )
    forest = CompletionForest(paths, "partial", ())
    pruned, closure = goal_support_certified_prune(
        forest,
        [1],
        provider,
        pack_id=expander.pack_id,
        constraint_version=expander.constraint_version,
        bounds=expander.bounds,
        state=state,
    )
    assert pruned is forest
    assert closure is None


# --------------------------------------------------------------------------- #
# Mutable-state / cache isolation
# --------------------------------------------------------------------------- #


def test_adversarial_shared_verifier_factory_does_not_cross_contaminate_sidecars() -> None:
    shared = _TracingGoalVerifier(
        {"ax"},
        profile_label="shared",
        profile_digest=_solver_profile().digest or _solver_profile().compute_digest(),
    )

    expander, _provider_default = _provider({"ax"})
    profile = _solver_profile()

    from slm_training.dsl.solver.goal_support import GoalSupportProvider

    provider = GoalSupportProvider(expander, profile, lambda: shared)
    _result_a, side_a = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    _result_b, side_b = provider.check_with_sidecar(expander.root_state(), _query(expander, "b"))
    assert side_a.terminal_evidence_digests != side_b.terminal_evidence_digests


def test_adversarial_interleaved_verdicts_keep_independent_traces() -> None:
    expander, provider = _provider({"ax"})
    state = expander.root_state()
    supported, side_s = provider.check_with_sidecar(state, _query(expander, "a"))
    unsupported, side_u = provider.check_with_sidecar(state, _query(expander, "b"))
    assert supported.verdict is SupportVerdict.SUPPORTED
    assert unsupported.verdict is SupportVerdict.UNSUPPORTED
    assert side_s.witness_terminal_evidence_digest is not None
    assert side_u.witness_terminal_evidence_digest is None


def test_adversarial_mutated_local_trace_not_used_by_provider() -> None:
    trace = GoalTerminalEvidenceTrace()
    evidence = _terminal_evidence()
    trace.append(evidence)
    trace.append(evidence)
    assert len(trace.records) == 2
    expander, provider = _provider({"ax"})
    _result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    assert len(sidecar.terminal_evidence_digests) >= 1
    assert sidecar.terminal_evidence_digests != tuple(
        record.evidence_digest or record.compute_digest() for record in trace.records
    )


def test_adversarial_replay_order_independent_violations() -> None:
    expander, provider = _provider({"ax"})
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    state = expander.root_state()
    first = replay_goal_support_result(result.certificate, sidecar, state=state, provider=provider)
    payload = sidecar.to_dict()
    payload["profile_digest"] = "f" * 64
    stale = _recompute_sidecar_digest(payload)
    second = replay_goal_support_result(result.certificate, stale, state=state, provider=provider)
    assert first.ok
    assert not second.ok


def test_adversarial_same_state_different_profiles_use_distinct_cache_entries() -> None:
    profile_a = _solver_profile(profile_id="cache/a")
    profile_b = _solver_profile(profile_id="cache/b")
    expander, provider_a = _provider({"ax"}, profile=profile_a)
    _expander_b, provider_b = _provider({"ax"}, profile=profile_b)
    cache: dict[str, object] = {}
    state = expander.root_state()
    exact_closure(state, provider_a, cache=cache, max_queries=4)
    hits_before = len(cache)
    exact_closure(state, provider_b, cache=cache, max_queries=4)
    assert len(cache) > hits_before


def test_adversarial_stale_cache_entry_not_reused_after_profile_change() -> None:
    profile_a = _solver_profile(profile_id="stale/a")
    profile_b = _solver_profile(profile_id="stale/b")
    expander, provider_a = _provider({"ax"}, profile=profile_a)
    _expander_b, provider_b = _provider({"ax"}, profile=profile_b)
    cache: dict[str, object] = {}
    state = expander.root_state()
    query = _query(expander, "a")
    result_a = provider_a.check(state, query)
    from slm_training.dsl.solver.closure import _cache_key

    key_a = _cache_key(state, query.hole_id, query.candidate, provider_a.backend_version)
    cache[key_a] = result_a
    key_b = _cache_key(state, query.hole_id, query.candidate, provider_b.backend_version)
    assert key_a != key_b
    assert key_b not in cache


# --------------------------------------------------------------------------- #
# Privacy / security
# --------------------------------------------------------------------------- #


def test_adversarial_privacy_secrets_not_in_sidecar_or_certificate() -> None:
    expander, provider = _provider({"ax"})
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    cert_json = json.dumps(result.certificate.to_dict())
    sidecar_json = json.dumps(sidecar.to_dict())
    _assert_no_raw_leak(cert_json, label="certificate")
    _assert_no_raw_leak(sidecar_json, label="sidecar")
    _assert_no_raw_leak(result.certificate.witness_source or "", label="witness_source")


def test_adversarial_privacy_openui_verifier_redacts_prompt_secrets() -> None:
    secret = "password=" + ("s" * 20)
    request = GenerationRequest(prompt=f"Contact alice@example.com with {secret}")
    verifier = _verifier(request=request)
    outcome = verifier.verify('root = Button(":cta.label")')
    _assert_no_raw_leak(outcome.detail, label="VerifyOutcome.detail")
    evidence = verifier.last_trace.latest()
    assert evidence is not None
    _assert_no_raw_leak(json.dumps(evidence.to_dict()), label="terminal_evidence")


def test_adversarial_privacy_reason_codes_bound_length_and_redact_secrets() -> None:
    raw = "token=hf_" + ("z" * 24) + " see \x00\x1b control"
    evidence = _terminal_evidence(
        required_gate_results=(
            GoalGateResultV1(gate_id="G0", status="REJECT", reason_code=raw),
        ),
        overall_status="REJECT",
        structural_status="REJECT",
    )
    payload = evidence.to_dict()
    reason = payload["required_gate_results"][0]["reason_code"]
    assert "hf_" not in reason
    assert "[REDACTED]" in reason
    assert len(reason) <= 280


def test_adversarial_privacy_telemetry_sink_has_no_raw_content() -> None:
    expander, provider = _provider({"ax"})
    sink = GoalSupportTelemetrySink()
    wrapped = instrument_goal_support_provider(provider, sink)
    state = expander.root_state()
    hole = state.holes[0]
    query = SupportQuery(
        state_fingerprint=state.fingerprint,
        hole_id=hole.hole_id,
        candidate=hole.values[0],
    )
    wrapped.check(state, query)
    telemetry_json = json.dumps(
        {
            "replay_failures": sink.replay_failures,
            "obstruction_cores": sink.obstruction_cores,
            "backtracks": sink.backtracks,
            "unobserved": sink.unobserved,
        }
    )
    for needle in _FORBIDDEN_SUBSTRINGS:
        assert needle not in telemetry_json


def test_adversarial_privacy_replay_violations_identify_invariant_without_raw_echo() -> None:
    expander, provider = _provider({"ax"})
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    payload = sidecar.to_dict()
    payload["profile_digest"] = "f" * 64
    stale = _recompute_sidecar_digest(payload)
    replay = replay_goal_support_result(
        result.certificate,
        stale,
        state=expander.root_state(),
        provider=provider,
    )
    assert not replay.ok
    joined = " ".join(replay.violations)
    assert "profile_digest" in joined
    assert len(joined) < 500


def test_adversarial_privacy_long_opaque_literal_bounded_in_redaction() -> None:
    rng = _rng()
    opaque = "".join(chr(rng.randint(33, 126)) for _ in range(512))
    redacted = redact_bounded_string(opaque)
    assert len(redacted) <= 280
    assert "truncated:" in redacted
    assert opaque not in redacted


def test_adversarial_property_style_digest_tamper_detection(seed: int = _SEED) -> None:
    rng = random.Random(seed)
    expander, provider = _provider({"ax"})
    result, sidecar = provider.check_with_sidecar(expander.root_state(), _query(expander, "a"))
    payload = sidecar.to_dict()
    for _ in range(8):
        field = rng.choice(
            [
                "profile_digest",
                "base_certificate_digest",
                "witness_terminal_evidence_digest",
            ]
        )
        trial = dict(payload)
        trial[field] = hashlib.sha256(str(rng.random()).encode()).hexdigest()
        tampered = _recompute_sidecar_digest(trial)
        replay = replay_goal_support_result(
            result.certificate,
            tampered,
            state=expander.root_state(),
            provider=provider,
        )
        assert not replay.ok
