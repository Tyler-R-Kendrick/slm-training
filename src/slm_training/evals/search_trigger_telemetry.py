"""EFS2-02 wiring: observe-only trigger telemetry before enabling recovery.

Provides a torch-free harness that records ``SearchTriggerObservationV1`` events
during compiler/search decoding without allowing the trigger to alter generation.
The observer can be attached to greedy, temperature-sampled, beam, or X22-style
valid-state decode regimes; the output is a labeled event stream plus firing-rate
statistics.

This module is eval-only wiring.  It loads no checkpoint, runs no model, and
makes no quality or ship claim.  It reuses ``StagnationTracker`` from
``slm_training.dsl.grammar.fastpath.lattice_search`` for repeated-state detection
but does not mutate the live ``LatticeSearchState``.

Invariants:

1. Observation never changes the decode trajectory, RNG state, or selected
   output; disabling telemetry yields byte-identical results.
2. ``BOTTOM``/hard conflict is recorded as a retraction event, never as a
   stochastic trajectory trigger.
3. ``STAGNATION`` is computed only from repeated finite-state fingerprints or
   lack of certified/decision progress.
4. ``UNCERTAINTY`` is computed only over the current finite legal-action set.
5. Outcome labels are computed after the run; they never enter the live trigger
   decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from slm_training.dsl.grammar.fastpath.lattice_search import StagnationTracker
from slm_training.versioning import UNKNOWN, build_version_stamp

TRIGGER_SCHEMA_VERSION = 1


class TriggerPredicate(str, Enum):
    """Kind of observable event that may lead to recovery."""

    BOTTOM = "BOTTOM"
    STAGNATION = "STAGNATION"
    UNCERTAINTY = "UNCERTAINTY"
    BUDGET_PRESSURE = "BUDGET_PRESSURE"


class TriggerRegime(str, Enum):
    """Decode regime under which the trigger was observed."""

    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    BEAM = "beam"
    X22 = "x22"


@dataclass(frozen=True)
class TriggerThresholdManifest:
    """Validation-selected thresholds, frozen before test analysis."""

    repeat_window: int = 3
    no_progress_window: int = 4
    margin_quantile: float = 0.1
    entropy_quantile: float = 0.75
    value_plateau_window: int = 3
    budget_pressure_forward_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeat_window": self.repeat_window,
            "no_progress_window": self.no_progress_window,
            "margin_quantile": self.margin_quantile,
            "entropy_quantile": self.entropy_quantile,
            "value_plateau_window": self.value_plateau_window,
            "budget_pressure_forward_limit": self.budget_pressure_forward_limit,
        }


@dataclass(frozen=True)
class SearchTriggerObservationV1:
    """One observe-only trigger observation at a finite decision step."""

    regime: TriggerRegime
    predicate: TriggerPredicate
    step_index: int
    state_fingerprint: str
    decision_depth: int
    trail_fingerprint: str
    live_action_count: int
    certified_reductions_since_prior: int
    repeated_state_count: int
    top1_score: float
    top2_score: float
    margin: float
    entropy: float
    value_score: float | None
    value_delta: float | None
    verifier_calls_since_progress: int
    model_forwards_since_progress: int
    wall_ms_since_progress: float
    pending_conflict_reason: str | None
    triggered: bool
    outcome_final_pass: bool | None = None
    outcome_recoverable: bool | None = None
    outcome_remaining_cost: float | None = None
    schema_version: int = TRIGGER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = {
            "regime": self.regime.value,
            "predicate": self.predicate.value,
            "step_index": self.step_index,
            "state_fingerprint": self.state_fingerprint,
            "decision_depth": self.decision_depth,
            "trail_fingerprint": self.trail_fingerprint,
            "live_action_count": self.live_action_count,
            "certified_reductions_since_prior": self.certified_reductions_since_prior,
            "repeated_state_count": self.repeated_state_count,
            "top1_score": _safe_float(self.top1_score),
            "top2_score": _safe_float(self.top2_score),
            "margin": _safe_float(self.margin),
            "entropy": _safe_float(self.entropy),
            "value_score": _safe_float(self.value_score),
            "value_delta": _safe_float(self.value_delta),
            "verifier_calls_since_progress": self.verifier_calls_since_progress,
            "model_forwards_since_progress": self.model_forwards_since_progress,
            "wall_ms_since_progress": _safe_float(self.wall_ms_since_progress),
            "pending_conflict_reason": self.pending_conflict_reason,
            "triggered": self.triggered,
            "outcome_final_pass": self.outcome_final_pass,
            "outcome_recoverable": self.outcome_recoverable,
            "outcome_remaining_cost": _safe_float(self.outcome_remaining_cost),
            "schema_version": self.schema_version,
        }
        return data


@dataclass(frozen=True)
class DecisionStep:
    """Synthetic decision step supplied by the caller or a decoder shim."""

    state_fingerprint: str
    decision_depth: int
    live_action_scores: tuple[float, ...]
    certified_reductions: int = 0
    value_score: float | None = None
    verifier_calls: int = 0
    model_forwards: int = 0
    wall_ms: float = 0.0
    pending_conflict_reason: str | None = None
    is_bottom: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_fingerprint": self.state_fingerprint,
            "decision_depth": self.decision_depth,
            "live_action_scores": [_safe_float(s) for s in self.live_action_scores],
            "certified_reductions": self.certified_reductions,
            "value_score": _safe_float(self.value_score),
            "verifier_calls": self.verifier_calls,
            "model_forwards": self.model_forwards,
            "wall_ms": _safe_float(self.wall_ms),
            "pending_conflict_reason": self.pending_conflict_reason,
            "is_bottom": self.is_bottom,
        }


@dataclass
class TriggerRunResult:
    """Collected observations for one regime × example."""

    regime: TriggerRegime
    example_id: str
    thresholds: TriggerThresholdManifest
    observations: list[SearchTriggerObservationV1] = field(default_factory=list)

    def firing_rate(self) -> float:
        if not self.observations:
            return 0.0
        return sum(1 for o in self.observations if o.triggered) / len(self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "example_id": self.example_id,
            "thresholds": self.thresholds.to_dict(),
            "observations": [o.to_dict() for o in self.observations],
            "firing_rate": self.firing_rate(),
        }


@dataclass
class TriggerComparisonResult:
    """Container for the EFS2-02 regime comparison."""

    thresholds: TriggerThresholdManifest
    runs: list[TriggerRunResult] = field(default_factory=list)
    version_stamp: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds.to_dict(),
            "runs": [r.to_dict() for r in self.runs],
            "version_stamp": self.version_stamp,
        }


def _safe_float(x: float | None) -> float | None:
    if x is None:
        return None
    return None if not (isinstance(x, float) and x == x) else float(x)


def _trail_fingerprint(
    prior_trail: str,
    state_fingerprint: str,
    decision_depth: int,
) -> str:
    payload = {"prior": prior_trail, "state": state_fingerprint, "depth": decision_depth}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _margin_and_entropy(scores: tuple[float, ...]) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    sorted_scores = sorted(scores, reverse=True)
    top1 = sorted_scores[0]
    top2 = sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    margin = top1 - top2
    # Normalize scores to a soft distribution and compute entropy.
    exps = [math.exp(max(s - top1, -20.0)) for s in scores]
    total = sum(exps)
    if total <= 0:
        return margin, 0.0
    probs = [e / total for e in exps]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return margin, entropy


class TriggerObserver:
    """Observe-only trigger collector.

    The observer records candidate trigger events but never mutates the search
    state.  ``StagnationTracker`` is used for repeated-state detection; all
    other predicates are computed from the supplied ``DecisionStep``.
    """

    def __init__(
        self,
        regime: TriggerRegime,
        thresholds: TriggerThresholdManifest,
        *,
        example_id: str = "",
    ) -> None:
        self.regime = regime
        self.thresholds = thresholds
        self.example_id = example_id
        self.result = TriggerRunResult(regime=regime, example_id=example_id, thresholds=thresholds)
        self._stagnation = StagnationTracker(patience=max(1, thresholds.repeat_window - 1))
        self._progress_step = 0
        self._last_value: float | None = None
        self._value_history: list[float] = []
        self._seen_states: dict[str, int] = {}
        self._trail = "seed"

    def observe(self, step_index: int, step: DecisionStep) -> SearchTriggerObservationV1 | None:
        """Record one step.  Returns the observation; the caller's state is unchanged."""
        margin, entropy = _margin_and_entropy(step.live_action_scores)
        self._seen_states[step.state_fingerprint] = self._seen_states.get(step.state_fingerprint, 0) + 1
        repeated_count = self._seen_states[step.state_fingerprint] - 1

        # Stagnation tracker treats (signature, progress) pairs; progress is
        # certified reductions + decision depth so both forms of movement count.
        progress = step.certified_reductions + step.decision_depth
        stagnated = self._stagnation.observe(step.state_fingerprint, progress)

        value_delta = None
        if step.value_score is not None:
            if self._last_value is not None:
                value_delta = step.value_score - self._last_value
            self._last_value = step.value_score
            self._value_history.append(step.value_score)

        self._trail = _trail_fingerprint(self._trail, step.state_fingerprint, step.decision_depth)

        # Determine the most salient predicate for this step.
        if step.is_bottom:
            predicate = TriggerPredicate.BOTTOM
        elif stagnated:
            predicate = TriggerPredicate.STAGNATION
        elif entropy > self.thresholds.entropy_quantile and margin < self.thresholds.margin_quantile:
            predicate = TriggerPredicate.UNCERTAINTY
        elif (
            self.thresholds.budget_pressure_forward_limit is not None
            and step.model_forwards > self.thresholds.budget_pressure_forward_limit
        ):
            predicate = TriggerPredicate.BUDGET_PRESSURE
        else:
            predicate = TriggerPredicate.UNCERTAINTY

        triggered = (
            step.is_bottom
            or stagnated
            or (entropy > self.thresholds.entropy_quantile and margin < self.thresholds.margin_quantile)
            or (
                self.thresholds.budget_pressure_forward_limit is not None
                and step.model_forwards > self.thresholds.budget_pressure_forward_limit
            )
        )

        observation = SearchTriggerObservationV1(
            regime=self.regime,
            predicate=predicate,
            step_index=step_index,
            state_fingerprint=step.state_fingerprint,
            decision_depth=step.decision_depth,
            trail_fingerprint=self._trail,
            live_action_count=len(step.live_action_scores),
            certified_reductions_since_prior=step.certified_reductions,
            repeated_state_count=repeated_count,
            top1_score=max(step.live_action_scores) if step.live_action_scores else 0.0,
            top2_score=sorted(step.live_action_scores, reverse=True)[1] if len(step.live_action_scores) > 1 else 0.0,
            margin=margin,
            entropy=entropy,
            value_score=step.value_score,
            value_delta=value_delta,
            verifier_calls_since_progress=step.verifier_calls,
            model_forwards_since_progress=step.model_forwards,
            wall_ms_since_progress=step.wall_ms,
            pending_conflict_reason=step.pending_conflict_reason,
            triggered=triggered,
        )
        self.result.observations.append(observation)
        return observation

    def label_outcomes(
        self,
        final_pass: bool,
        recoverable: bool,
        remaining_cost: float | None = None,
    ) -> None:
        """Attach after-run outcome labels to every observation.

        Labels are the same for all steps in this wiring fixture; a real
        implementation would label per-step from an offline oracle.
        """
        self.result.observations = [
            SearchTriggerObservationV1(
                regime=o.regime,
                predicate=o.predicate,
                step_index=o.step_index,
                state_fingerprint=o.state_fingerprint,
                decision_depth=o.decision_depth,
                trail_fingerprint=o.trail_fingerprint,
                live_action_count=o.live_action_count,
                certified_reductions_since_prior=o.certified_reductions_since_prior,
                repeated_state_count=o.repeated_state_count,
                top1_score=o.top1_score,
                top2_score=o.top2_score,
                margin=o.margin,
                entropy=o.entropy,
                value_score=o.value_score,
                value_delta=o.value_delta,
                verifier_calls_since_progress=o.verifier_calls_since_progress,
                model_forwards_since_progress=o.model_forwards_since_progress,
                wall_ms_since_progress=o.wall_ms_since_progress,
                pending_conflict_reason=o.pending_conflict_reason,
                triggered=o.triggered,
                outcome_final_pass=final_pass,
                outcome_recoverable=recoverable,
                outcome_remaining_cost=remaining_cost,
            )
            for o in self.result.observations
        ]


def compare_trigger_regimes(
    examples: Iterable[tuple[str, Iterable[DecisionStep], bool, bool]],
    thresholds: TriggerThresholdManifest | None = None,
    regimes: tuple[TriggerRegime, ...] = (
        TriggerRegime.GREEDY,
        TriggerRegime.TEMPERATURE,
        TriggerRegime.BEAM,
    ),
    *,
    seed: int = 2026,
    stamp_components: tuple[str, ...] = ("evals.scoring",),
) -> TriggerComparisonResult:
    """Run observe-only trigger telemetry across regimes and examples.

    ``examples`` is an iterable of (example_id, steps, final_pass, recoverable).
    For each regime the same steps are observed with a regime-specific score
    perturbation (temperature/beam noise) to simulate non-greedy decode.
    """
    thresholds = thresholds or TriggerThresholdManifest()
    result = TriggerComparisonResult(thresholds=thresholds)
    rng = random.Random(seed)
    for example_id, steps, final_pass, recoverable in examples:
        base_steps = list(steps)
        for regime in regimes:
            observer = TriggerObserver(regime, thresholds, example_id=example_id)
            for i, step in enumerate(base_steps):
                perturbed_scores = tuple(step.live_action_scores)
                if regime is TriggerRegime.TEMPERATURE:
                    # Add small temperature-like noise without changing the order much.
                    perturbed_scores = tuple(
                        max(0.0, s + rng.gauss(0.0, 0.05)) for s in step.live_action_scores
                    )
                elif regime is TriggerRegime.BEAM:
                    # Beam slightly reorders by small random tie-breaks.
                    perturbed_scores = tuple(
                        s + rng.uniform(-0.01, 0.01) for s in step.live_action_scores
                    )
                observer.observe(
                    i,
                    DecisionStep(
                        state_fingerprint=step.state_fingerprint,
                        decision_depth=step.decision_depth,
                        live_action_scores=perturbed_scores,
                        certified_reductions=step.certified_reductions,
                        value_score=step.value_score,
                        verifier_calls=step.verifier_calls,
                        model_forwards=step.model_forwards,
                        wall_ms=step.wall_ms,
                        pending_conflict_reason=step.pending_conflict_reason,
                        is_bottom=step.is_bottom,
                    ),
                )
            observer.label_outcomes(final_pass, recoverable)
            result.runs.append(observer.result)
    try:
        result.version_stamp = build_version_stamp(*stamp_components)
    except KeyError:
        result.version_stamp = {
            "stamp_schema": UNKNOWN,
            "components": {cid: UNKNOWN for cid in stamp_components},
            "note": "version stamp unavailable",
        }
    return result
