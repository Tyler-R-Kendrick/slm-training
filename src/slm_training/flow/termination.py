"""Termination policy protocol and reference arms for flow and direct-policy samplers.

SLM-191 (FFE2-03): specify and ablate STOP, total-hazard, and fixed-K termination
semantics on exact CTMC fixtures.  No trained model is required here; the arms
consume scalar signals (STOP score, total hazard, absorption probability,
selector probability, oracle edit count) that a production sampler would emit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from slm_training.flow.reference.generator import Generator
from slm_training.flow.reference.trajectory import FlowTrajectoryV1


STOP = "STOP"
HOLD = "HOLD"
ABSTAIN = "ABSTAIN"

STOP_EDIT = "STOP_EDIT"
TOTAL_HAZARD = "TOTAL_HAZARD"
ABSORB = "ABSORB"
FIXED_K_END = "FIXED_K_END"
SELECTOR_END = "SELECTOR_END"
HYBRID_END = "HYBRID_END"
ORACLE_LENGTH = "ORACLE_LENGTH"
MAX_STEPS = "MAX_STEPS"
NO_LIVE_CANDIDATES = "NO_LIVE_CANDIDATES"
UNKNOWN_BUDGET = "UNKNOWN_BUDGET"


@dataclass(frozen=True)
class TerminationContext:
    """Everything a termination policy is allowed to observe at one step."""

    state_fingerprint: str
    step_index: int = 0
    edit_count: int = 0
    wall_time: float = 0.0
    total_hazard: float | None = None
    stop_score: float | None = None
    absorption_prob: float | None = None
    selector_prob: float | None = None
    oracle_edit_count: int | None = None
    candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_fingerprint": self.state_fingerprint,
            "step_index": self.step_index,
            "edit_count": self.edit_count,
            "wall_time": self.wall_time,
            "total_hazard": self.total_hazard,
            "stop_score": self.stop_score,
            "absorption_prob": self.absorption_prob,
            "selector_prob": self.selector_prob,
            "oracle_edit_count": self.oracle_edit_count,
            "candidates": list(self.candidates),
        }


@dataclass(frozen=True)
class TerminationDecision:
    """Decision returned by a termination policy."""

    action: str
    reason: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@runtime_checkable
class TerminationPolicy(Protocol):
    """Protocol shared by direct-policy and flow samplers for stopping."""

    @property
    def name(self) -> str: ...

    def decide(self, ctx: TerminationContext) -> TerminationDecision: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ExplicitStopPolicy:
    """Stop when the model's STOP score exceeds a threshold."""

    stop_threshold: float = 0.5
    max_steps: int = 20
    name: str = "explicit_stop"

    def decide(self, ctx: TerminationContext) -> TerminationDecision:
        score = ctx.stop_score
        if score is not None and score >= self.stop_threshold:
            return TerminationDecision(STOP, STOP_EDIT, confidence=float(score))
        if ctx.step_index >= self.max_steps:
            return TerminationDecision(STOP, MAX_STEPS, confidence=1.0)
        return TerminationDecision(HOLD, "", confidence=1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stop_threshold": self.stop_threshold,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class AbsorbingHazardPolicy:
    """Stop when total hazard is negligible or absorption is very likely."""

    hazard_threshold: float = 1e-6
    absorb_threshold: float = 0.9
    max_steps: int = 20
    name: str = "absorbing_hazard"

    def decide(self, ctx: TerminationContext) -> TerminationDecision:
        hazard = ctx.total_hazard
        if hazard is not None and hazard <= self.hazard_threshold:
            conf = 1.0 - min(1.0, hazard / max(self.hazard_threshold, 1e-12))
            return TerminationDecision(STOP, TOTAL_HAZARD, confidence=float(conf))
        absorb = ctx.absorption_prob
        if absorb is not None and absorb >= self.absorb_threshold:
            return TerminationDecision(STOP, ABSORB, confidence=float(absorb))
        if ctx.step_index >= self.max_steps:
            return TerminationDecision(STOP, MAX_STEPS, confidence=1.0)
        return TerminationDecision(HOLD, "", confidence=1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hazard_threshold": self.hazard_threshold,
            "absorb_threshold": self.absorb_threshold,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class FixedKPolicy:
    """Stop after a fixed number of edits."""

    k: int = 4
    max_steps: int = 20
    name: str = "fixed_k"

    def decide(self, ctx: TerminationContext) -> TerminationDecision:
        if ctx.edit_count >= self.k:
            return TerminationDecision(STOP, FIXED_K_END, confidence=1.0)
        if ctx.step_index >= self.max_steps:
            return TerminationDecision(STOP, MAX_STEPS, confidence=1.0)
        return TerminationDecision(HOLD, "", confidence=1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "k": self.k,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class FixedKPlusSelectorPolicy:
    """Stop after K edits only when a selector head also agrees."""

    k: int = 4
    selector_threshold: float = 0.5
    max_steps: int = 20
    name: str = "fixed_k_plus_selector"

    def decide(self, ctx: TerminationContext) -> TerminationDecision:
        selector = ctx.selector_prob
        if ctx.edit_count >= self.k and selector is not None and selector >= self.selector_threshold:
            return TerminationDecision(STOP, SELECTOR_END, confidence=float(selector))
        if ctx.step_index >= self.max_steps:
            return TerminationDecision(STOP, MAX_STEPS, confidence=1.0)
        return TerminationDecision(HOLD, "", confidence=1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "k": self.k,
            "selector_threshold": self.selector_threshold,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class HybridMinProgressPolicy:
    """Stop on explicit STOP or on K edits + selector support."""

    min_k: int = 2
    stop_threshold: float = 0.5
    selector_threshold: float = 0.5
    max_steps: int = 20
    name: str = "hybrid_min_progress"

    def decide(self, ctx: TerminationContext) -> TerminationDecision:
        score = ctx.stop_score
        if score is not None and score >= self.stop_threshold:
            return TerminationDecision(STOP, HYBRID_END, confidence=float(score))
        selector = ctx.selector_prob
        if (
            ctx.edit_count >= self.min_k
            and selector is not None
            and selector >= self.selector_threshold
        ):
            return TerminationDecision(STOP, HYBRID_END, confidence=float(selector))
        if ctx.step_index >= self.max_steps:
            return TerminationDecision(STOP, MAX_STEPS, confidence=1.0)
        return TerminationDecision(HOLD, "", confidence=1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_k": self.min_k,
            "stop_threshold": self.stop_threshold,
            "selector_threshold": self.selector_threshold,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class OracleLengthPolicy:
    """Oracle baseline: stop at the known minimum edit distance to target."""

    oracle_edit_count: int | None = None
    max_steps: int = 20
    name: str = "oracle_length"

    def decide(self, ctx: TerminationContext) -> TerminationDecision:
        oracle = self.oracle_edit_count if self.oracle_edit_count is not None else ctx.oracle_edit_count
        if oracle is not None and ctx.edit_count >= oracle:
            return TerminationDecision(STOP, ORACLE_LENGTH, confidence=1.0)
        if ctx.step_index >= self.max_steps:
            return TerminationDecision(STOP, MAX_STEPS, confidence=1.0)
        return TerminationDecision(HOLD, "", confidence=1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "oracle_edit_count": self.oracle_edit_count,
            "max_steps": self.max_steps,
        }


POLICY_REGISTRY: dict[str, type[TerminationPolicy]] = {
    "explicit_stop": ExplicitStopPolicy,
    "absorbing_hazard": AbsorbingHazardPolicy,
    "fixed_k": FixedKPolicy,
    "fixed_k_plus_selector": FixedKPlusSelectorPolicy,
    "hybrid_min_progress": HybridMinProgressPolicy,
    "oracle_length": OracleLengthPolicy,
}


def build_termination_policy(name: str, **kwargs: Any) -> TerminationPolicy:
    """Construct a registered termination policy by name."""
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown termination policy {name!r}; choose from {sorted(POLICY_REGISTRY)}"
        )
    return POLICY_REGISTRY[name](**kwargs)  # type: ignore[return-value]


def _candidate_fingerprints(generator: Generator, state_fingerprint: str) -> tuple[str, ...]:
    idx = generator.state_index.get(state_fingerprint)
    if idx is None:
        return ()
    successors = generator.legal_successors(idx)
    return tuple(generator.index_state[j].fingerprint for j, _, _ in successors)


def sample_with_termination(
    generator: Generator,
    source: Any,
    policy: TerminationPolicy,
    terminal_check: Any,
    rng: Any,
    *,
    max_wall_time: float = 1e6,
    oracle_edit_count: int | None = None,
    stop_score_fn: Any | None = None,
    absorption_prob_fn: Any | None = None,
    selector_prob_fn: Any | None = None,
) -> tuple[FlowTrajectoryV1, dict[str, Any]]:
    """Sample one CTMC path governed by a termination policy.

    The signal functions receive the current state object and return floats in
    [0, 1] (or None).  They are fixtures here; a production sampler replaces
    them with model heads.
    """
    states: list[Any] = [source]
    actions: list[str] = []
    holding_times: list[float] = []
    wall_times: list[float] = [0.0]
    certificates: list[str] = []
    decisions: list[TerminationDecision] = []
    visited_valid: set[str] = {source.fingerprint}
    current = source
    total_time = 0.0
    stop_reason = UNKNOWN_BUDGET

    for step in range(policy.max_steps):
        if terminal_check(current):
            stop_reason = "terminal_state"
            break

        candidates = _candidate_fingerprints(generator, current.fingerprint)
        ctx = TerminationContext(
            state_fingerprint=current.fingerprint,
            step_index=step,
            edit_count=len(actions),
            wall_time=total_time,
            total_hazard=generator.hazard(current),
            stop_score=stop_score_fn(current) if stop_score_fn is not None else None,
            absorption_prob=absorption_prob_fn(current) if absorption_prob_fn is not None else None,
            selector_prob=selector_prob_fn(current) if selector_prob_fn is not None else None,
            oracle_edit_count=oracle_edit_count,
            candidates=candidates,
        )
        decision = policy.decide(ctx)
        decisions.append(decision)
        if decision.action == ABSTAIN:
            stop_reason = "abstained"
            break
        if decision.action == STOP:
            stop_reason = decision.reason
            break

        successors = generator.legal_successors(current)
        if not successors:
            stop_reason = NO_LIVE_CANDIDATES
            break
        total_rate = sum(rate for _, _, rate in successors)
        if total_rate <= 0.0:
            stop_reason = NO_LIVE_CANDIDATES
            break
        hold_time = rng.expovariate(total_rate)
        if total_time + hold_time > max_wall_time:
            stop_reason = MAX_STEPS
            break

        threshold = rng.random() * total_rate
        cumsum = 0.0
        chosen_idx = successors[0][0]
        chosen_action = successors[0][1]
        for idx, action, rate in successors:
            cumsum += rate
            if cumsum >= threshold:
                chosen_idx = idx
                chosen_action = action
                break

        action_id = chosen_action.action_id if chosen_action is not None else ""
        cert_id = f"{current.fingerprint[:12]}->{generator.index_state[chosen_idx].fingerprint[:12]}"
        actions.append(action_id)
        holding_times.append(hold_time)
        total_time += hold_time
        wall_times.append(total_time)
        certificates.append(cert_id)
        current = generator.index_state[chosen_idx]
        states.append(current)
        visited_valid.add(current.fingerprint)

    trajectory = FlowTrajectoryV1(
        trajectory_id=f"{policy.name}-{rng.randint(0, 2**31 - 1)}",
        source_fingerprint=source.fingerprint,
        states=tuple(s.fingerprint for s in states),
        actions=tuple(actions),
        holding_times=tuple(holding_times),
        wall_times=tuple(wall_times),
        certificates=tuple(certificates),
        terminal_fingerprint=states[-1].fingerprint,
        total_time=total_time,
    )
    metadata = {
        "stop_reason": stop_reason,
        "decisions": [d.to_dict() for d in decisions],
        "visited_valid_states": len(visited_valid),
        "abstained": stop_reason == "abstained",
    }
    return trajectory, metadata


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def brier_score(predicted: Sequence[float], observed: Sequence[int]) -> float:
    """Mean squared difference between predicted probabilities and binary outcomes."""
    if not predicted or not observed or len(predicted) != len(observed):
        return 0.0
    return sum((_clamp(p) - float(o)) ** 2 for p, o in zip(predicted, observed)) / len(predicted)


def expected_calibration_error(
    predicted: Sequence[float], observed: Sequence[int], n_bins: int = 5
) -> float:
    """ECE with uniform probability bins."""
    if not predicted or not observed or len(predicted) != len(observed):
        return 0.0
    pairs = list(zip(predicted, observed))
    if not pairs:
        return 0.0
    total = len(pairs)
    ece = 0.0
    for bin_index in range(n_bins):
        low = bin_index / n_bins
        high = (bin_index + 1) / n_bins
        in_bin = [(p, o) for p, o in pairs if low <= _clamp(p) < high or (bin_index == n_bins - 1 and _clamp(p) == 1.0)]
        if not in_bin:
            continue
        avg_pred = sum(p for p, _ in in_bin) / len(in_bin)
        avg_obs = sum(o for _, o in in_bin) / len(in_bin)
        ece += abs(avg_pred - avg_obs) * (len(in_bin) / total)
    return ece


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance between discrete distributions."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


__all__ = [
    "STOP",
    "HOLD",
    "ABSTAIN",
    "STOP_EDIT",
    "TOTAL_HAZARD",
    "ABSORB",
    "FIXED_K_END",
    "SELECTOR_END",
    "HYBRID_END",
    "ORACLE_LENGTH",
    "MAX_STEPS",
    "NO_LIVE_CANDIDATES",
    "UNKNOWN_BUDGET",
    "TerminationContext",
    "TerminationDecision",
    "TerminationPolicy",
    "ExplicitStopPolicy",
    "AbsorbingHazardPolicy",
    "FixedKPolicy",
    "FixedKPlusSelectorPolicy",
    "HybridMinProgressPolicy",
    "OracleLengthPolicy",
    "build_termination_policy",
    "sample_with_termination",
    "brier_score",
    "expected_calibration_error",
    "total_variation",
]
