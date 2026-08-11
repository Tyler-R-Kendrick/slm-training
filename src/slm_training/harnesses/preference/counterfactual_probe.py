"""LDI3-03 verifier-backed counterfactual action-value probe (SLM-131).

Orchestration for turning delayed OpenUI failures into defensible *same-state*
action evidence: admit candidate states, force compiler-legal alternatives at the
exact stored state, roll each forward under common seeds, retain the full ordered
verifier vector, and derive pure value materializers (Pareto / lexicographic /
scalar / binary) plus a semantic good/bad partition.

PGS-D01 (SLM-502) adds a **distinct** support-backed path:
:class:`GoalSupportProbeInputs` + :func:`run_goal_support_probe` query bounded
goal support per legal action and emit ``GoalActionEvidenceV1`` /
``GoalDomainAdequacyReportV1``. That path is absolute existential/exhaustive
evidence under a pinned profile — it does **not** alias, overwrite, or equate to
the relative ``semantic_partition`` rollout probe above.

This layer is model-free: the actual rollout + G0-G12 verification live behind the
:class:`RolloutBackend` protocol (the model plug-in "only forces an action and
produces an outcome"). The orchestration — admission, legal-action selection,
identity-caching, common-seed comparison, resume/dedup, and the value/verdict
derivations — is pure and deterministic. No model training runs here, delayed
failures never become token labels from final-output location alone, and partial
evidence stays unresolved.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from slm_training.dsl.solver.goal_support import (
    GOAL_SUPPORT_IMPLEMENTATION_VERSION,
    GoalActionEvidenceV1,
    GoalActionWorkCountersV1,
    GoalDomainActionPartitionsV1,
    GoalDomainAdequacyReportV1,
    GoalDomainAdequacyStatsV1,
    GoalSupportProvider,
    GoalSupportResultV1,
    action_id_from_value,
    bounds_digest_from_state,
    legal_set_fingerprint,
    replay_goal_support_result,
)
from slm_training.dsl.solver.goal_support_domain_adequacy import (
    _aggregate_obstruction_summary,
    _assign_partition,
    _classify_domain_adequacy,
    _terminal_records,
)
from slm_training.dsl.solver.state import DomainValue, FiniteDomainState, HoleId
from slm_training.dsl.solver.support import (
    EnumerativeSupportOracle,
    SearchCounters,
    SupportQuery,
)
from slm_training.harnesses.preference.decision_events_v2 import (
    ActionOutcomeV2,
    DecisionStateV2,
)

__all__ = [
    "AdmissionReason",
    "CandidateState",
    "GoalSupportProbeConfig",
    "GoalSupportProbeInputs",
    "ProbeConfig",
    "RawOutcome",
    "RolloutBackend",
    "admit_states",
    "goal_support_cache_key",
    "goal_support_probe_cap_policy_table",
    "run_goal_support_probe",
    "select_actions",
    "select_goal_support_actions",
    "outcome_cache_key",
    "run_probe",
    "pareto_front",
    "lexicographic_key",
    "scalar_value",
    "binary_verdict",
    "semantic_partition",
]

# Admission sources. ``heuristic_only`` (final-output blame) can never become
# semantic training data — it is quarantined to a diagnostic queue.
AdmissionReason = Literal[
    "immediate_verifier_failure",
    "detector_localized",
    "constraint_shadow_probe",
    "small_margin_policy",
    "incomplete_action_evidence",
    "heuristic_only",
]
_SEMANTIC_REASONS: frozenset[str] = frozenset(
    {
        "immediate_verifier_failure",
        "detector_localized",
        "constraint_shadow_probe",
        "small_margin_policy",
        "incomplete_action_evidence",
    }
)


@dataclass(frozen=True)
class CandidateState:
    state: DecisionStateV2
    reason: AdmissionReason
    priority: float = 0.0


@dataclass(frozen=True)
class ProbeConfig:
    action_cap: int = 8
    min_rollouts: int = 3
    min_effect: float = 0.1
    seeds: tuple[int, ...] = (0, 1, 2)
    required_gates: tuple[str, ...] = ("G0", "G1")  # hard gates a "good" action may not worsen
    allow_heuristic: bool = False
    max_states: int | None = None

    def __post_init__(self) -> None:
        if self.action_cap <= 0 or self.min_rollouts <= 0:
            raise ValueError("action_cap and min_rollouts must be positive")
        if not self.seeds:
            raise ValueError("at least one continuation seed is required")


@dataclass(frozen=True)
class RawOutcome:
    """What a rollout of one forced action under one seed produced. ``verifier_vector``
    is the complete ordered G0-G12 mapping; ``resolved`` is False on timeout/error/
    partial evidence (kept but never treated as a clean result)."""

    canonical_output: str | None
    finish_reason: str
    verifier_vector: tuple[tuple[str, str], ...]
    resolved: bool = True
    judge_evidence: tuple[tuple[str, str], ...] = ()
    reward_vector: tuple[tuple[str, float], ...] = ()


class RolloutBackend(Protocol):
    """Forces ``action_id`` at the exact ``state`` and continues under ``seed``.
    The only model-dependent surface; deferred to a GPU-backed run."""

    def rollout(self, state: DecisionStateV2, action_id: int, seed: int) -> RawOutcome: ...


def admit_states(
    candidates: Sequence[CandidateState], *, allow_heuristic: bool = False
) -> tuple[list[CandidateState], list[CandidateState]]:
    """Split candidates into an admitted probe queue and a rejected list.

    Heuristic (final-output-blame) candidates are rejected unless explicitly
    requested, and even then are quarantined — they never enter the semantic queue.
    The admitted queue is deterministically ordered by ``(-priority, state_id)``.
    """
    admitted: list[CandidateState] = []
    rejected: list[CandidateState] = []
    for cand in candidates:
        if cand.reason == "heuristic_only" and not allow_heuristic:
            rejected.append(cand)
        else:
            admitted.append(cand)
    admitted.sort(key=lambda c: (-c.priority, c.state.state_id))
    return admitted, rejected


def select_actions(
    state: DecisionStateV2,
    *,
    policy_action: int,
    policy_probs: Mapping[int, float] | None = None,
    cap: int = 8,
    roles: Mapping[int, str] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Choose actions to probe from the exact legal set (never global token top-k).

    Always includes the policy action; includes all legal actions when the set is
    within ``cap``; otherwise selects deterministically by policy probability, then
    semantic-role coverage, then action id. Returns ``(selected, excluded)``.
    """
    legal = tuple(state.legal_action_ids)
    if policy_action not in legal:
        raise ValueError("policy_action must be within the state's legal set")
    if len(legal) <= cap:
        return legal, ()
    probs = policy_probs or {}
    roles = roles or {}
    seen_roles: set[str] = set()

    def rank(a: int) -> tuple[float, float, int]:
        # higher prob first; then reward covering a new role; then lower id
        role_bonus = 0.0 if roles.get(a, "") in seen_roles else -1.0
        return (-probs.get(a, 0.0), role_bonus, a)

    ordered = sorted((a for a in legal if a != policy_action), key=rank)
    selected = [policy_action]
    for a in ordered:
        if len(selected) >= cap:
            break
        selected.append(a)
        seen_roles.add(roles.get(a, ""))
    selected_set = set(selected)
    excluded = tuple(a for a in legal if a not in selected_set)
    return tuple(sorted(selected)), excluded


def outcome_cache_key(
    state_id: str,
    action_id: int,
    seed: int,
    *,
    policy_sha: str,
    decoder_hash: str,
    verifier_hash: str,
) -> str:
    """Content identity for one rollout. A changed policy/decoder/verifier identity
    yields a different key, so incompatible outcomes are never reused."""
    payload = json.dumps(
        [state_id, action_id, seed, policy_sha, decoder_hash, verifier_hash],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _outcome_hash(raw: RawOutcome) -> str:
    payload = json.dumps(
        [raw.canonical_output, raw.finish_reason, list(raw.verifier_vector), raw.resolved],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_probe(
    admitted: Sequence[CandidateState],
    backend: RolloutBackend,
    *,
    config: ProbeConfig,
    selection: Mapping[str, Sequence[int]] | None = None,
    cache: dict[str, RawOutcome] | None = None,
) -> list[ActionOutcomeV2]:
    """Roll forced actions forward under the common seed set, caching by identity.

    Resumable: a populated ``cache`` short-circuits already-computed rollouts, so an
    interrupted run reproduces the same final manifest. Aggregates per
    ``(state, action)`` into one :class:`ActionOutcomeV2` with all seeds/vectors.
    """
    cache = cache if cache is not None else {}
    states = admitted if config.max_states is None else admitted[: config.max_states]
    outcomes: list[ActionOutcomeV2] = []
    for cand in states:
        state = cand.state
        actions = (
            tuple(selection[state.state_id])
            if selection and state.state_id in selection
            else state.legal_action_ids
        )
        for action_id in actions:
            seeds: list[int] = []
            out_hashes: list[str] = []
            vectors: list[tuple[tuple[str, str], ...]] = []
            rewards: list[tuple[tuple[str, float], ...]] = []
            for seed in config.seeds:
                key = outcome_cache_key(
                    state.state_id, action_id, seed,
                    policy_sha=state.policy_checkpoint_sha,
                    decoder_hash=state.decode_config_hash,
                    verifier_hash=state.verifier_bundle_hash,
                )
                raw = cache.get(key)
                if raw is None:
                    raw = backend.rollout(state, action_id, seed)
                    cache[key] = raw
                seeds.append(seed)
                out_hashes.append(_outcome_hash(raw))
                vectors.append(raw.verifier_vector)
                if raw.reward_vector:
                    rewards.append(raw.reward_vector)
            outcomes.append(
                ActionOutcomeV2(
                    state_id=state.state_id,
                    action_id=action_id,
                    legal=action_id in set(state.legal_action_ids),
                    rollout_policy_sha=state.policy_checkpoint_sha,
                    continuation_seeds=tuple(seeds),
                    outcome_hashes=tuple(out_hashes),
                    verifier_vectors=tuple(vectors),
                    reward_vectors=tuple(rewards),
                    evidence_confidence=1.0 if len(seeds) >= config.min_rollouts else 0.0,
                )
            )
    return outcomes


# --- pure value / verdict materializers -----------------------------------


def _gate_pass(vector: Mapping[str, str], gate: str) -> bool:
    return vector.get(gate, "fail") == "pass"


def _mean_gate_pass(outcome: ActionOutcomeV2, gate: str) -> float:
    if not outcome.verifier_vectors:
        return 0.0
    passes = sum(1 for v in outcome.verifier_vectors if _gate_pass(dict(v), gate))
    return passes / len(outcome.verifier_vectors)


def pareto_front(
    outcomes: Sequence[ActionOutcomeV2], metrics: Sequence[str]
) -> list[int]:
    """Action ids on the Pareto front over mean gate-pass rates (higher better)."""
    scored = {o.action_id: [_mean_gate_pass(o, m) for m in metrics] for o in outcomes}
    front: list[int] = []
    for a, sa in scored.items():
        dominated = any(
            b != a
            and all(sb_i >= sa_i for sb_i, sa_i in zip(sb, sa))
            and any(sb_i > sa_i for sb_i, sa_i in zip(sb, sa))
            for b, sb in scored.items()
        )
        if not dominated:
            front.append(a)
    return sorted(front)


def lexicographic_key(outcome: ActionOutcomeV2, priorities: Sequence[str]) -> tuple[float, ...]:
    """Sort key (descending preference) over gate priorities, e.g. valid → meaningful."""
    return tuple(_mean_gate_pass(outcome, g) for g in priorities)


def scalar_value(outcome: ActionOutcomeV2, weights: Mapping[str, float]) -> float:
    return sum(w * _mean_gate_pass(outcome, g) for g, w in weights.items())


def binary_verdict(
    outcome: ActionOutcomeV2, required_gates: Sequence[str]
) -> Literal["pass", "fail", "unresolved"]:
    """Pass/fail only when every required gate is observed in every rollout;
    otherwise unresolved (partial/insufficient evidence)."""
    if not outcome.verifier_vectors:
        return "unresolved"
    for vector in outcome.verifier_vectors:
        vec = dict(vector)
        if any(g not in vec for g in required_gates):
            return "unresolved"
        if any(vec[g] != "pass" for g in required_gates):
            return "fail"
    return "pass"


@dataclass(frozen=True)
class SemanticPartition:
    good_action_ids: tuple[int, ...]
    bad_action_ids: tuple[int, ...]
    ambiguous_action_ids: tuple[int, ...]
    unobserved_action_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "good_action_ids": list(self.good_action_ids),
            "bad_action_ids": list(self.bad_action_ids),
            "ambiguous_action_ids": list(self.ambiguous_action_ids),
            "unobserved_action_ids": list(self.unobserved_action_ids),
        }


def semantic_partition(
    state: DecisionStateV2,
    outcomes: Sequence[ActionOutcomeV2],
    *,
    config: ProbeConfig,
    policy_action: int,
    value_gate: str = "G1",
) -> SemanticPartition:
    """Derive a good/bad/ambiguous partition under the strict same-state rules.

    Values are measured against the *policy action* baseline (the counterfactual
    reference). An action is *good* only with enough rollouts, a value margin over
    that baseline of at least ``min_effect``, and no required-hard-gate regression;
    *bad* with a symmetric negative margin. Insufficient evidence → ambiguous; a
    legal action never probed → unobserved. Delayed-failure blame is never inferred
    from final-output position — only from these forced-action rollout verdicts.
    """
    observed = {o.action_id: o for o in outcomes if o.state_id == state.state_id}
    baseline = (
        _mean_gate_pass(observed[policy_action], value_gate)
        if policy_action in observed
        else 0.0
    )
    good: list[int] = []
    bad: list[int] = []
    ambiguous: list[int] = []
    for action_id, outcome in observed.items():
        n = len(outcome.continuation_seeds)
        if n < config.min_rollouts:
            ambiguous.append(action_id)
            continue
        value = _mean_gate_pass(outcome, value_gate)
        worsens_hard = any(
            _mean_gate_pass(outcome, g) < baseline - 1e-9 for g in config.required_gates
        )
        if value >= baseline + config.min_effect and not worsens_hard:
            good.append(action_id)
        elif value <= baseline - config.min_effect:
            bad.append(action_id)
        else:
            ambiguous.append(action_id)
    unobserved = [a for a in state.legal_action_ids if a not in observed]
    return SemanticPartition(
        tuple(sorted(good)), tuple(sorted(bad)), tuple(sorted(ambiguous)), tuple(sorted(unobserved))
    )


# --- PGS-D01 goal-support probe (distinct from relative rollout partition) ---


@dataclass(frozen=True)
class GoalSupportProbeConfig:
    """Bounded goal-support query policy for one exact ``DecisionStateV2``."""

    exact_action_cap: int = 8

    def __post_init__(self) -> None:
        if self.exact_action_cap <= 0:
            raise ValueError("exact_action_cap must be positive")


@dataclass(frozen=True)
class GoalSupportProbeInputs:
    """Frozen bridge from ``DecisionStateV2`` to bounded goal-support queries."""

    decision_state: DecisionStateV2
    domain_state: FiniteDomainState
    hole_id: HoleId
    policy_action: int
    action_values: Mapping[int, DomainValue]
    provider: GoalSupportProvider
    config: GoalSupportProbeConfig = GoalSupportProbeConfig()
    policy_probs: Mapping[int, float] | None = None
    action_roles: Mapping[int, str] | None = None


def goal_support_probe_cap_policy_table() -> dict[str, Any]:
    """Policy-aware exact action-cap selection (counterfactual-probe owner).

    Distinct from ``domain_adequacy_cap_policy_table`` lexicographic subset in
    ``analyze_goal_domain``: this path always retains the policy-selected legal
    action and ranks the remainder by policy probability, role coverage, then id
    — mirroring :func:`select_actions`.
    """
    return {
        "within_cap": "query every legal action through bounded VSS goal support",
        "above_cap": {
            "selection": (
                "policy-selected action plus deterministic fill by policy "
                "probability, then semantic-role coverage, then action id "
                "(same rank as select_actions)"
            ),
            "always_include_policy_action": True,
            "omitted_partition": "unobserved",
            "inference_forbidden": True,
        },
        "greedy_continuation_forbidden": True,
        "grammar_legality_independent": True,
        "semantic_partition_alias_forbidden": True,
    }


def select_goal_support_actions(
    decision_state: DecisionStateV2,
    action_values: Mapping[int, DomainValue],
    *,
    policy_action: int,
    cap: int,
    policy_probs: Mapping[int, float] | None = None,
    roles: Mapping[int, str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Choose opaque goal-support action ids to query from the legal int set.

    Uses :func:`select_actions` on compiler ``legal_action_ids`` and maps each
    chosen int to its ``action_id_from_value`` digest. Returns
    ``(query_action_ids, unobserved_action_ids, cap_applied)``.
    """
    int_selected, int_excluded = select_actions(
        decision_state,
        policy_action=policy_action,
        policy_probs=policy_probs,
        cap=cap,
        roles=roles,
    )
    query_ids = tuple(
        sorted(action_id_from_value(action_values[a]) for a in int_selected)
    )
    unobserved_ids = tuple(
        sorted(action_id_from_value(action_values[a]) for a in int_excluded)
    )
    return query_ids, unobserved_ids, bool(unobserved_ids)


def goal_support_cache_key(
    *,
    state_fingerprint: str,
    action_id: str,
    profile_digest: str,
    problem_id: str,
    pack_id: str,
    constraint_version: str,
    bounds_digest: str,
    legal_set_fingerprint: str,
    decode_config_hash: str,
    implementation_version: str = GOAL_SUPPORT_IMPLEMENTATION_VERSION,
) -> str:
    """Content identity for one goal-support action query.

    A changed state, action, profile, problem, constraint, pack/bounds, decoder,
    or implementation version yields a different key — incompatible evidence is
    never reused.
    """
    payload = json.dumps(
        [
            state_fingerprint,
            action_id,
            profile_digest,
            problem_id,
            pack_id,
            constraint_version,
            bounds_digest,
            legal_set_fingerprint,
            decode_config_hash,
            implementation_version,
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_cached_goal_evidence(
    row: GoalActionEvidenceV1 | None,
    *,
    state: FiniteDomainState,
    hole_id: HoleId,
    action_id: str,
    value: DomainValue,
    legal_fp: str,
    profile_digest: str,
    provider: GoalSupportProvider,
    hard_profile: bool,
    observed: bool,
) -> tuple[GoalActionEvidenceV1, tuple[str, ...], SearchCounters] | None:
    """Re-bind and replay cached proof rows; untrusted cache input never labels."""
    if row is None or not row.evidence_digest:
        return None
    try:
        bound = GoalActionEvidenceV1.from_dict(row.to_dict(include_digest=True))
    except (TypeError, ValueError):
        return None
    if (
        bound.state_fingerprint != state.fingerprint
        or bound.legal_set_fingerprint != legal_fp
        or bound.profile_digest != profile_digest
        or bound.hole_id != hole_id.to_dict()
        or bound.action_id != action_id
        or bound.authority_tier != provider.profile.authority_tier
        or bound.legal is not True
    ):
        return None
    if not observed:
        return (
            (bound, (), SearchCounters())
            if bound.partition == "unobserved"
            else None
        )
    if bound.partition == "unobserved":
        return None
    try:
        certificate = provider.certificate(bound.base_certificate_digest)
        sidecar = provider.goal_result(bound.base_certificate_digest)
    except LookupError:
        return None
    if (
        certificate.query.state_fingerprint != state.fingerprint
        or certificate.query.hole_id != hole_id
        or action_id_from_value(certificate.query.candidate) != action_id
        or certificate.query.candidate != value
        or sidecar.digest != bound.goal_sidecar_digest
    ):
        return None
    replay = provider.replay(certificate, state=state)
    expected = _assign_partition(
        observed=True,
        verdict=certificate.verdict,
        replay_ok=replay.ok,
        hard_profile=hard_profile,
    )
    if not replay.ok or bound.partition != expected or not bound.replay_ok:
        return None
    core_atoms: tuple[str, ...] = ()
    core = provider.obstruction_core(bound.base_certificate_digest)
    if bound.obstruction_core_digest is not None:
        if core is None or (core.core_digest or core.compute_digest()) != (
            bound.obstruction_core_digest
        ):
            return None
        core_atoms = core.core_atoms
    return bound, core_atoms, replay.counters


def _validate_goal_support_inputs(inputs: GoalSupportProbeInputs) -> None:
    legal = set(inputs.decision_state.legal_action_ids)
    if inputs.policy_action not in legal:
        raise ValueError("policy_action must be within the state's legal set")
    if set(inputs.action_values.keys()) != legal:
        raise ValueError("action_values must cover exactly the legal action set")
    domain = inputs.domain_state.domain(inputs.hole_id)
    domain_by_id = {action_id_from_value(value): value for value in domain.values}
    for int_id, value in inputs.action_values.items():
        digest = action_id_from_value(value)
        if digest not in domain_by_id:
            raise ValueError(f"action_values[{int_id}] is absent from the domain hole")


def _unobserved_goal_evidence(
    *,
    state_fingerprint: str,
    legal_fp: str,
    profile_digest: str,
    hole_id: HoleId,
    action_id: str,
    authority_tier: str,
) -> GoalActionEvidenceV1:
    row = GoalActionEvidenceV1(
        state_fingerprint=state_fingerprint,
        legal_set_fingerprint=legal_fp,
        profile_digest=profile_digest,
        hole_id=hole_id.to_dict(),
        action_id=action_id,
        partition="unobserved",
        authority_tier=authority_tier,  # type: ignore[arg-type]
    )
    return GoalActionEvidenceV1.from_dict(row.to_dict(include_digest=True))


def _query_goal_action_evidence(
    *,
    state: FiniteDomainState,
    hole_id: HoleId,
    action_id: str,
    value: DomainValue,
    provider: GoalSupportProvider,
    legal_fp: str,
    profile_digest: str,
    hard_profile: bool,
) -> tuple[GoalActionEvidenceV1, tuple[str, ...], int, int, int, int, str | None]:
    """Return evidence row plus work counters and optional obstruction core atoms."""
    query = SupportQuery(
        state_fingerprint=state.fingerprint,
        hole_id=hole_id,
        candidate=value,
    )
    result, sidecar = provider.check_with_sidecar(state, query)
    replay = replay_goal_support_result(
        result.certificate,
        sidecar,
        state=state,
        provider=provider,
    )
    replay_ok = replay.ok
    partition = _assign_partition(
        observed=True,
        verdict=result.verdict,
        replay_ok=replay_ok,
        hard_profile=hard_profile,
    )
    terminal_count = len(sidecar.terminal_evidence_digests)
    nodes = result.counters.nodes + replay.counters.nodes
    tokens = result.counters.tokens + replay.counters.tokens
    depth = max(result.counters.depth, replay.counters.depth)
    backtracks = result.counters.backtracks + replay.counters.backtracks
    verifier_calls = result.counters.verifier_calls + replay.counters.verifier_calls

    core_atoms: tuple[str, ...] = ()
    if (
        partition == "unsupported"
        and replay_ok
        and sidecar.obstruction_core_digest
    ):
        from slm_training.dsl.solver.goal_support_obstruction import (
            compute_domain_obstruction_core,
        )

        replay_verifier = provider._fresh_verifier()
        core_result = EnumerativeSupportOracle(
            provider.expander, replay_verifier
        ).check(state, query)
        nodes += core_result.counters.nodes
        tokens += core_result.counters.tokens
        depth = max(depth, core_result.counters.depth)
        backtracks += core_result.counters.backtracks
        verifier_calls += core_result.counters.verifier_calls
        core = compute_domain_obstruction_core(
            certificate=result.certificate,
            terminal_records=_terminal_records(replay_verifier),
            profile=provider.profile,
            state=state,
            replay_ok=True,
        )
        if core is not None:
            core_atoms = core.core_atoms

    sidecar_bound = GoalSupportResultV1.from_dict(sidecar.to_dict(include_digest=True))
    row = GoalActionEvidenceV1(
        state_fingerprint=state.fingerprint,
        legal_set_fingerprint=legal_fp,
        profile_digest=profile_digest,
        hole_id=hole_id.to_dict(),
        action_id=action_id,
        partition=partition,
        base_certificate_digest=result.certificate.digest,
        goal_sidecar_digest=sidecar_bound.digest or sidecar_bound.compute_digest(),
        authority_tier=provider.profile.authority_tier,
        replay_ok=replay_ok,
        work_counters=GoalActionWorkCountersV1(
            nodes=nodes,
            tokens=tokens,
            depth=depth,
            backtracks=backtracks,
            verifier_calls=verifier_calls,
            terminal_count=terminal_count,
        ),
        obstruction_core_digest=(
            sidecar.obstruction_core_digest
            if partition == "unsupported" and replay_ok
            else None
        ),
        stop_reason=result.certificate.stop_reason,
    )
    bound = GoalActionEvidenceV1.from_dict(row.to_dict(include_digest=True))

    return (
        bound,
        core_atoms,
        nodes,
        backtracks,
        verifier_calls,
        terminal_count,
        result.certificate.stop_reason,
    )


def run_goal_support_probe(
    inputs: GoalSupportProbeInputs,
    *,
    cache: dict[str, GoalActionEvidenceV1] | None = None,
) -> tuple[GoalDomainAdequacyReportV1, tuple[GoalActionEvidenceV1, ...]]:
    """Query bounded goal support for selected legal actions; classify adequacy.

    Resumable: a populated ``cache`` keyed by :func:`goal_support_cache_key`
    short-circuits already-computed action evidence. This path is model-free
    except for receiving the frozen exact state and policy-selected action; it
    never aliases or overwrites relative rollout ``semantic_partition`` evidence.
    """
    _validate_goal_support_inputs(inputs)
    cache = cache if cache is not None else {}
    state = inputs.domain_state
    hole_id = inputs.hole_id
    domain_coverage = str(
        dict(state.domain(hole_id).metadata).get("coverage", "complete")
    )
    if domain_coverage not in {"complete", "partial", "none"}:
        raise ValueError("domain coverage must be complete, partial, or none")
    provider = inputs.provider
    provider.validate_identities(state)

    profile = provider.profile
    profile_digest = provider.profile_digest
    hard_profile = profile.authority_tier in frozenset(
        {"compiler-hard", "verifier-hard"}
    )

    value_by_int = dict(inputs.action_values)
    value_by_id = {
        action_id_from_value(value): value for value in value_by_int.values()
    }
    action_ids = tuple(sorted(value_by_id))
    selected_action_id = action_id_from_value(value_by_int[inputs.policy_action])

    legal_fp = legal_set_fingerprint(
        state_fingerprint=state.fingerprint,
        hole_id=hole_id,
        action_ids=action_ids,
    )
    bounds_digest = bounds_digest_from_state(state)
    decode_hash = inputs.decision_state.decode_config_hash

    query_ids, unobserved_ids, cap_applied = select_goal_support_actions(
        inputs.decision_state,
        inputs.action_values,
        policy_action=inputs.policy_action,
        cap=inputs.config.exact_action_cap,
        policy_probs=inputs.policy_probs,
        roles=inputs.action_roles,
    )

    evidence_rows: list[GoalActionEvidenceV1] = []
    supported: list[str] = []
    unsupported: list[str] = []
    unknown: list[str] = []
    unobserved: list[str] = list(unobserved_ids)

    total_terminals = 0
    total_verifier_calls = 0
    total_nodes = 0
    total_backtracks = 0
    stop_reasons: set[str] = set()
    all_unsupported_replay_valid = True
    obstruction_core_atoms: set[str] = set()

    for action_id in query_ids:
        key = goal_support_cache_key(
            state_fingerprint=state.fingerprint,
            action_id=action_id,
            profile_digest=profile_digest,
            problem_id=state.problem_id,
            pack_id=state.pack_id,
            constraint_version=state.constraint_version,
            bounds_digest=bounds_digest,
            legal_set_fingerprint=legal_fp,
            decode_config_hash=decode_hash,
        )
        cached = _validated_cached_goal_evidence(
            cache.get(key),
            state=state,
            hole_id=hole_id,
            action_id=action_id,
            value=value_by_id[action_id],
            legal_fp=legal_fp,
            profile_digest=profile_digest,
            provider=provider,
            hard_profile=hard_profile,
            observed=True,
        )
        if cached is None:
            row, core_atoms, nodes, backtracks, verifier_calls, terminals, stop = (
                _query_goal_action_evidence(
                    state=state,
                    hole_id=hole_id,
                    action_id=action_id,
                    value=value_by_id[action_id],
                    provider=provider,
                    legal_fp=legal_fp,
                    profile_digest=profile_digest,
                    hard_profile=hard_profile,
                )
            )
            cache[key] = row
            total_terminals += terminals
            total_verifier_calls += verifier_calls
            total_nodes += nodes
            total_backtracks += backtracks
            if stop:
                stop_reasons.add(stop)
            obstruction_core_atoms.update(core_atoms)
        else:
            row, core_atoms, replay_counters = cached
            total_terminals += row.work_counters.terminal_count
            total_verifier_calls += (
                row.work_counters.verifier_calls + replay_counters.verifier_calls
            )
            total_nodes += row.work_counters.nodes + replay_counters.nodes
            total_backtracks += (
                row.work_counters.backtracks + replay_counters.backtracks
            )
            if row.stop_reason:
                stop_reasons.add(row.stop_reason)
            obstruction_core_atoms.update(core_atoms)

        partition = row.partition
        if partition == "supported":
            supported.append(action_id)
        elif partition == "unsupported":
            unsupported.append(action_id)
            if not row.replay_ok:
                all_unsupported_replay_valid = False
        elif partition == "unknown":
            unknown.append(action_id)
            all_unsupported_replay_valid = False

        evidence_rows.append(row)

    for action_id in unobserved_ids:
        key = goal_support_cache_key(
            state_fingerprint=state.fingerprint,
            action_id=action_id,
            profile_digest=profile_digest,
            problem_id=state.problem_id,
            pack_id=state.pack_id,
            constraint_version=state.constraint_version,
            bounds_digest=bounds_digest,
            legal_set_fingerprint=legal_fp,
            decode_config_hash=decode_hash,
        )
        cached = _validated_cached_goal_evidence(
            cache.get(key),
            state=state,
            hole_id=hole_id,
            action_id=action_id,
            value=value_by_id[action_id],
            legal_fp=legal_fp,
            profile_digest=profile_digest,
            provider=provider,
            hard_profile=hard_profile,
            observed=False,
        )
        if cached is None:
            row = _unobserved_goal_evidence(
                state_fingerprint=state.fingerprint,
                legal_fp=legal_fp,
                profile_digest=profile_digest,
                hole_id=hole_id,
                action_id=action_id,
                authority_tier=profile.authority_tier,
            )
            cache[key] = row
        else:
            row, _core_atoms, _replay_counters = cached
        evidence_rows.append(row)

    partitions = GoalDomainActionPartitionsV1(
        supported=tuple(sorted(supported)),
        unsupported=tuple(sorted(unsupported)),
        unknown=tuple(sorted(unknown)),
        unobserved=tuple(sorted(unobserved)),
    )
    evidence_tuple = tuple(sorted(evidence_rows, key=lambda item: item.action_id))
    candidate_summary = _aggregate_obstruction_summary(
        evidence_tuple,
        classification="domain_inadequate_under_bounds",
        hard_profile=hard_profile,
        core_atoms_union=tuple(sorted(obstruction_core_atoms)),
    )
    classification = _classify_domain_adequacy(
        partitions,
        selected_action_id,
        all_unsupported_replay_valid=all_unsupported_replay_valid,
        hard_profile=hard_profile,
        cap_applied=cap_applied,
        domain_complete=domain_coverage == "complete",
        obstruction_summary=candidate_summary,
    )
    obstruction_summary = (
        candidate_summary
        if classification == "domain_inadequate_under_bounds"
        else None
    )
    evidence_digests = tuple(
        sorted(item.evidence_digest or item.compute_digest() for item in evidence_tuple)
    )

    report = GoalDomainAdequacyReportV1(
        classification=classification,
        state_fingerprint=state.fingerprint,
        legal_set_fingerprint=legal_fp,
        selected_action_id=selected_action_id,
        profile_digest=profile_digest,
        problem_id=state.problem_id,
        pack_id=state.pack_id,
        constraint_version=state.constraint_version,
        bounds_digest=bounds_digest,
        hole_id=hole_id.to_dict(),
        partitions=partitions,
        evidence_digests=evidence_digests,
        stats=GoalDomainAdequacyStatsV1(
            action_count=len(action_ids),
            queried_action_count=len(query_ids),
            terminal_count=total_terminals,
            query_count=len(query_ids),
            verifier_calls=total_verifier_calls,
            expanded_nodes=total_nodes,
            backtracks=total_backtracks,
            stop_reasons=tuple(sorted(stop_reasons)),
        ),
        obstruction_summary=obstruction_summary,
        exact_action_cap=inputs.config.exact_action_cap,
        cap_applied=cap_applied,
        domain_coverage=domain_coverage,
    )
    bound = GoalDomainAdequacyReportV1.from_dict(report.to_dict(include_digest=True))
    return bound, evidence_tuple
