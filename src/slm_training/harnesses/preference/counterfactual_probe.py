"""LDI3-03 verifier-backed counterfactual action-value probe (SLM-131).

Orchestration for turning delayed OpenUI failures into defensible *same-state*
action evidence: admit candidate states, force compiler-legal alternatives at the
exact stored state, roll each forward under common seeds, retain the full ordered
verifier vector, and derive pure value materializers (Pareto / lexicographic /
scalar / binary) plus a semantic good/bad partition.

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

from slm_training.harnesses.preference.decision_events_v2 import (
    ActionOutcomeV2,
    DecisionStateV2,
)

__all__ = [
    "AdmissionReason",
    "CandidateState",
    "ProbeConfig",
    "RawOutcome",
    "RolloutBackend",
    "admit_states",
    "select_actions",
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
