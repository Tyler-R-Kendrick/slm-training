"""CAP4-02: compiler-floor + runtime-signal adaptive residual-plane routing.

This module schedules how many residual ternary planes to execute for a
grammar-state decision.  It combines a compiler-derived structural floor with
runtime posterior entropy, top-two margin, grammar-state sensitivity, or a
learned router.  Every schedule respects the floor and never delegates legal
membership to the router.

This is a wiring/reference implementation: it scores each plane prefix by
calling the residual stack repeatedly.  Real speed-up would require an
incremental packed-plane kernel; when that kernel is unavailable, plane-count
savings are reported separately from wall-clock results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn

from slm_training.harnesses.distill.grammar_trace import (
    compute_entropy,
    compute_margin,
    normalize_legal_probs,
)
from slm_training.models.quantization.residual_planes import PlaneOutput

if TYPE_CHECKING:
    from slm_training.models.local_action_head import (
        LocalActionOutput,
        ResidualTritPlaneHead,
    )


ScheduleMode = Literal[
    "uniform_1",
    "uniform_max",
    "structural_floor",
    "floor_plus_entropy",
    "floor_plus_margin",
    "floor_plus_sensitivity",
    "floor_plus_learned_router",
    "oracle_min_planes",
]


def _ternary_digits_needed(n: int) -> int:
    """Minimum base-3 digits (trits) to represent ``n`` distinct symbols."""
    if n <= 1:
        return 0
    return math.ceil(math.log(n, 3))


def local_action_floor(branch_count: int) -> int:
    """Structural floor for a one-step local action head.

    Forced states (``branch_count <= 1``) need zero model planes because the
    action is compiler-determined.
    """
    return _ternary_digits_needed(branch_count)


def completion_support_floor(support_size: int | None) -> int:
    """Structural floor from CAP1-04 posterior effective/credible support.

    ``support_size`` is the number of completions that carry meaningful posterior
    mass at the declared distortion.  When unavailable the floor defaults to 0.
    """
    if support_size is None or support_size <= 1:
        return 0
    return _ternary_digits_needed(support_size)


def margin_preservation_floor(branch_count: int) -> int:
    """Margin-preservation floor for balanced geometric planes.

    For the reference balanced-ternary schedule this equals the local-action
    code floor.  Learned independent scales cannot claim an analytic ``3^R``
    guarantee; they rely on empirical per-state error envelopes instead.
    """
    return local_action_floor(branch_count)


@dataclass(frozen=True)
class PlaneScheduleSpec:
    """Versioned spec for a residual-plane schedule.

    Attributes:
        schedule_id: human-readable identifier, also used to select presets.
        latent_role: which latent/score role this schedule serves.
        max_planes: hard upper bound on planes (usually ``head.R``).
        structural_floor: how the compiler-derived floor is computed.
        runtime_signal: which runtime signal modulates the floor.
        thresholds: schedule-specific thresholds (entropy, margin, sensitivity,
            router logit, etc.).
        grouping_policy: ``whole_batch`` or ``compact``.
        fallback_policy: action when the schedule reaches ``max_planes`` without
            a stable decision.
    """

    schedule_id: str
    latent_role: Literal["local_action", "plan_support", "energy_margin"]
    max_planes: int
    structural_floor: str
    runtime_signal: Literal[
        "none", "posterior_entropy", "top2_margin", "sensitivity", "learned_router"
    ]
    thresholds: Mapping[str, float]
    grouping_policy: str
    fallback_policy: str


@dataclass
class RuntimeDiagnostics:
    """Runtime confidence/residual diagnostics available to the scheduler."""

    entropy: float | None = None
    margin: float | None = None
    sensitivity: Mapping[str, float] | None = None
    residual_norm: float = 0.0


@dataclass
class AdaptivePlaneRouteResult:
    """Result of routing one state through the adaptive plane pipeline."""

    action_identity: str | None
    decision_kind: str
    confidence: float
    planes_used: int
    max_planes: int
    telemetry: dict[str, Any] = field(default_factory=dict)


def _mean_sensitivity(sensitivity: Mapping[str, float] | None) -> float:
    if not sensitivity:
        return 0.0
    values = [float(v) for v in sensitivity.values()]
    return sum(values) / len(values) if values else 0.0


def _diagnostics_from_logits(
    logits: torch.Tensor,
    sensitivity: Mapping[str, float] | None,
    residual_norm: float,
) -> RuntimeDiagnostics:
    """Build runtime diagnostics from per-legal-action logits."""
    # Per-item logits shape: [batch, num_legal].  Process each row separately
    # because legal action counts may differ.
    batch = logits.shape[0]
    if batch == 0:
        return RuntimeDiagnostics(residual_norm=residual_norm)
    values = logits[0].tolist()
    probs = normalize_legal_probs(values, convention="logit")
    return RuntimeDiagnostics(
        entropy=compute_entropy(probs),
        margin=compute_margin(values, selected_index=None, convention="logit"),
        sensitivity=sensitivity,
        residual_norm=residual_norm,
    )


class PlaneRouter(nn.Module):
    """Tiny MLP that predicts whether another plane is worth executing.

    Features are bounded summaries that do not include the target action or any
    future verification outcome: branch count, entropy, margin, residual norm,
    and a mean sensitivity summary.  The router advises the schedule; it does
    not alter legal membership.
    """

    def __init__(self, feature_dim: int = 5, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a scalar logit for "execute another plane"."""
        return self.net(features).squeeze(-1)

    @staticmethod
    def build_features(
        branch_count: int,
        diagnostics: RuntimeDiagnostics,
    ) -> torch.Tensor:
        """Construct a feature vector from compiler/runtime state."""
        return torch.tensor(
            [
                float(branch_count),
                diagnostics.entropy or 0.0,
                diagnostics.margin or 0.0,
                diagnostics.residual_norm,
                _mean_sensitivity(diagnostics.sensitivity),
            ],
            dtype=torch.float32,
        )


class PlaneScheduler:
    """Compute how many planes to execute for one decision state."""

    def __init__(
        self,
        spec: PlaneScheduleSpec,
        *,
        router: PlaneRouter | None = None,
    ) -> None:
        self.spec = spec
        self.router = router
        if spec.runtime_signal == "learned_router" and self.router is None:
            self.router = PlaneRouter()

    def floor_planes(self, *, branch_count: int = 0, support_size: int | None = None) -> int:
        """Compiler-derived minimum number of planes."""
        floor_id = self.spec.structural_floor
        if floor_id == "local_action_code":
            return local_action_floor(branch_count)
        if floor_id == "completion_support":
            return completion_support_floor(support_size)
        if floor_id == "margin_preservation":
            return margin_preservation_floor(branch_count)
        if floor_id == "zero":
            return 0
        raise ValueError(f"unknown structural_floor: {floor_id!r}")

    def _runtime_request(
        self,
        current_planes: int,
        diagnostics: RuntimeDiagnostics,
        branch_count: int,
    ) -> bool:
        """Return True if the runtime signal asks for one more plane."""
        signal = self.spec.runtime_signal
        thresholds = self.spec.thresholds
        if signal == "none":
            return False
        if signal == "posterior_entropy":
            high = thresholds.get("entropy_high")
            if high is None:
                return False
            return (diagnostics.entropy or 0.0) > high
        if signal == "top2_margin":
            low = thresholds.get("margin_low")
            if low is None:
                return False
            return (diagnostics.margin or float("inf")) < low
        if signal == "sensitivity":
            high = thresholds.get("sensitivity_high")
            if high is None:
                return False
            return _mean_sensitivity(diagnostics.sensitivity) > high
        if signal == "learned_router":
            if self.router is None:
                return False
            features = PlaneRouter.build_features(branch_count, diagnostics)
            logit = float(self.router(features).item())
            return logit > thresholds.get("router_logit", 0.0)
        raise ValueError(f"unknown runtime_signal: {signal!r}")

    def desired_planes(
        self,
        current_planes: int,
        diagnostics: RuntimeDiagnostics,
        *,
        branch_count: int = 0,
        support_size: int | None = None,
    ) -> int:
        """Total planes requested after considering floor and runtime signal."""
        floor = self.floor_planes(branch_count=branch_count, support_size=support_size)
        if self.spec.schedule_id == "uniform_1":
            return max(1, floor)
        if self.spec.schedule_id == "uniform_max":
            return self.spec.max_planes
        if self.spec.schedule_id == "structural_floor":
            return floor
        if self.spec.schedule_id == "oracle_min_planes":
            # Oracle needs the full prefix sweep; the routing context handles it.
            return self.spec.max_planes
        # floor_plus_*: request another plane while the runtime signal is active.
        base = max(current_planes, floor)
        if self._runtime_request(current_planes, diagnostics, branch_count):
            return min(base + 1, self.spec.max_planes)
        return base


def _decode_head(
    head: "ResidualTritPlaneHead",
    output: "LocalActionOutput",
    legal_actions: list[str],
) -> tuple[str | None, str, float]:
    """Decode a head output into (action_identity, decision_kind, confidence)."""
    from slm_training.models.local_action_head import LocalActionOutput

    if not isinstance(output, LocalActionOutput):
        raise TypeError("expected LocalActionOutput")
    decision = head.decode(output, legal_actions)
    return (
        decision.action_identity,
        decision.decision_kind,
        decision.confidence,
    )


class AdaptivePlaneRoutingContext:
    """Execute an adaptive-plane schedule for a batch of grammar decisions."""

    def __init__(
        self,
        head: "ResidualTritPlaneHead",
        scheduler: PlaneScheduler,
        *,
        grouping_policy: str = "whole_batch",
        stability_patience: int = 1,
    ) -> None:
        from slm_training.models.local_action_head import ResidualTritPlaneHead

        if not isinstance(head, ResidualTritPlaneHead):
            raise TypeError("AdaptivePlaneRoutingContext requires ResidualTritPlaneHead")
        self.head = head
        self.scheduler = scheduler
        self.grouping_policy = grouping_policy
        self.stability_patience = stability_patience
        self.max_planes = scheduler.spec.max_planes

    def _score(
        self,
        hidden: torch.Tensor,
        state_context: Any,
        legal_actions: list[str],
        max_planes: int,
    ) -> tuple["LocalActionOutput", PlaneOutput]:
        """Score with ``max_planes`` and return both logits and plane diagnostics."""
        output = self.head.score(
            hidden,
            state_context,
            legal_actions,
            max_planes=max_planes,
            return_diagnostics=True,
        )
        diag = output.metadata.get("plane_diagnostics")
        if diag is None:
            # Fallback if diagnostics were not emitted.
            diag = self.head.residual_stack(
                hidden,
                max_planes=max_planes,
                return_diagnostics=True,
            )
        return output, diag

    def _item_diagnostics(
        self,
        logits: torch.Tensor,
        state_context: Any,
        plane_diagnostics: PlaneOutput,
    ) -> RuntimeDiagnostics:
        """Build diagnostics for a single row of a batch."""
        residual_norms = plane_diagnostics.residual_norms
        residual_norm = sum(residual_norms) / len(residual_norms) if residual_norms else 0.0
        sensitivity = getattr(state_context, "sensitivity", None)
        return _diagnostics_from_logits(logits, sensitivity, residual_norm)

    def route_batch(
        self,
        hidden: torch.Tensor,
        state_contexts: Sequence[Any],
        legal_actions_list: Sequence[list[str]],
    ) -> list[AdaptivePlaneRouteResult]:
        """Route a batch and return one result per input item."""
        batch_size = hidden.shape[0]
        assert len(state_contexts) == batch_size
        assert len(legal_actions_list) == batch_size

        results: list[AdaptivePlaneRouteResult | None] = [None] * batch_size
        active: list[int] = []

        for i in range(batch_size):
            ctx = state_contexts[i]
            legal = legal_actions_list[i]
            if getattr(ctx, "forced", False) or len(legal) <= 1:
                results[i] = AdaptivePlaneRouteResult(
                    action_identity=legal[0] if legal else None,
                    decision_kind="forced",
                    confidence=1.0,
                    planes_used=0,
                    max_planes=self.max_planes,
                    telemetry={"skip_reason": "forced_state"},
                )
            else:
                active.append(i)

        previous_actions: dict[int, str | None] = {i: None for i in active}
        stable_count: dict[int, int] = {i: 0 for i in active}

        for p in range(0, self.max_planes + 1):
            if not active:
                break

            if self.grouping_policy == "compact":
                indices = list(active)
            else:
                indices = [
                    i for i in range(batch_size)
                    if results[i] is None
                ]

            hidden_slice = hidden[indices]
            ctx_slice = [state_contexts[i] for i in indices]
            legal_slice = [legal_actions_list[i] for i in indices]

            # Score every active item at exactly ``p`` planes.  Items whose floor
            # is larger than ``p`` will keep running because desired_planes > p.
            outputs: list[LocalActionOutput] = []
            diags: list[PlaneOutput] = []
            for h, ctx, legal in zip(hidden_slice, ctx_slice, legal_slice):
                out, diag = self._score(
                    h.unsqueeze(0),
                    ctx,
                    legal,
                    max_planes=p,
                )
                outputs.append(out)
                diags.append(diag)

            for offset, i in enumerate(indices):
                ctx = ctx_slice[offset]
                legal = legal_slice[offset]
                out = outputs[offset]
                diag = diags[offset]
                logits = out.logits
                if logits is None:
                    continue
                action, decision_kind, confidence = _decode_head(
                    self.head, out, legal
                )

                diagnostics = self._item_diagnostics(logits, ctx, diag)
                branch_count = getattr(ctx, "branch_count", len(legal))
                support_size = getattr(ctx, "completion_support_size", None)

                desired = self.scheduler.desired_planes(
                    current_planes=p,
                    diagnostics=diagnostics,
                    branch_count=branch_count,
                    support_size=support_size,
                )

                stable = action == previous_actions[i]
                if stable:
                    stable_count[i] += 1
                else:
                    stable_count[i] = 0
                previous_actions[i] = action

                done = p >= desired and stable_count[i] >= self.stability_patience
                if done or p == self.max_planes:
                    telemetry = {
                        "final_action": action,
                        "decision_kind": decision_kind,
                        "confidence": confidence,
                        "entropy": diagnostics.entropy,
                        "margin": diagnostics.margin,
                        "residual_norm": diagnostics.residual_norm,
                        "sensitivity_mean": _mean_sensitivity(diagnostics.sensitivity),
                    }
                    # Fallback is triggered when we hit the hard cap while the
                    # schedule still wants another plane or while the decision is
                    # not yet stable.
                    fallback = p == self.max_planes and (
                        not done
                        or self.scheduler._runtime_request(
                            p, diagnostics, branch_count
                        )
                    )
                    if fallback:
                        telemetry["fallback_triggered"] = True
                        if self.scheduler.spec.fallback_policy == "abstain":
                            decision_kind = "abstain"
                            action = None
                    results[i] = AdaptivePlaneRouteResult(
                        action_identity=action,
                        decision_kind=decision_kind,
                        confidence=confidence,
                        planes_used=p,
                        max_planes=self.max_planes,
                        telemetry=telemetry,
                    )
                    if i in active:
                        active.remove(i)

        # Any items that did not finish (e.g. empty legal set) get an abstain.
        for i in range(batch_size):
            if results[i] is None:
                results[i] = AdaptivePlaneRouteResult(
                    action_identity=None,
                    decision_kind="abstain",
                    confidence=0.0,
                    planes_used=self.max_planes,
                    max_planes=self.max_planes,
                    telemetry={"reason": "routing_incomplete"},
                )

        return [r for r in results if r is not None]


# Convenience presets for the eight schedule modes required by CAP4-02.

SCHEDULE_PRESETS: dict[str, dict[str, Any]] = {
    "uniform_1": {
        "latent_role": "local_action",
        "structural_floor": "zero",
        "runtime_signal": "none",
        "thresholds": {},
        "grouping_policy": "whole_batch",
        "fallback_policy": "score_best",
    },
    "uniform_max": {
        "latent_role": "local_action",
        "structural_floor": "zero",
        "runtime_signal": "none",
        "thresholds": {},
        "grouping_policy": "whole_batch",
        "fallback_policy": "score_best",
    },
    "structural_floor": {
        "latent_role": "local_action",
        "structural_floor": "local_action_code",
        "runtime_signal": "none",
        "thresholds": {},
        "grouping_policy": "whole_batch",
        "fallback_policy": "score_best",
    },
    "floor_plus_entropy": {
        "latent_role": "local_action",
        "structural_floor": "local_action_code",
        "runtime_signal": "posterior_entropy",
        "thresholds": {"entropy_high": 0.5},
        "grouping_policy": "whole_batch",
        "fallback_policy": "score_best",
    },
    "floor_plus_margin": {
        "latent_role": "local_action",
        "structural_floor": "local_action_code",
        "runtime_signal": "top2_margin",
        "thresholds": {"margin_low": 0.2},
        "grouping_policy": "whole_batch",
        "fallback_policy": "score_best",
    },
    "floor_plus_sensitivity": {
        "latent_role": "local_action",
        "structural_floor": "local_action_code",
        "runtime_signal": "sensitivity",
        "thresholds": {"sensitivity_high": 0.5},
        "grouping_policy": "whole_batch",
        "fallback_policy": "score_best",
    },
    "floor_plus_learned_router": {
        "latent_role": "local_action",
        "structural_floor": "local_action_code",
        "runtime_signal": "learned_router",
        "thresholds": {"router_logit": 0.0},
        "grouping_policy": "compact",
        "fallback_policy": "score_best",
    },
    "oracle_min_planes": {
        "latent_role": "local_action",
        "structural_floor": "local_action_code",
        "runtime_signal": "none",
        "thresholds": {},
        "grouping_policy": "whole_batch",
        "fallback_policy": "score_best",
    },
}


def make_schedule_spec(
    schedule_id: ScheduleMode,
    max_planes: int,
    *,
    grouping_policy: str | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> PlaneScheduleSpec:
    """Build a ``PlaneScheduleSpec`` from a named CAP4-02 preset."""
    preset = SCHEDULE_PRESETS[schedule_id].copy()
    if grouping_policy is not None:
        preset["grouping_policy"] = grouping_policy
    if thresholds is not None:
        preset["thresholds"] = dict(thresholds)
    return PlaneScheduleSpec(
        schedule_id=schedule_id,
        latent_role=preset["latent_role"],
        max_planes=max_planes,
        structural_floor=preset["structural_floor"],
        runtime_signal=preset["runtime_signal"],
        thresholds=preset["thresholds"],
        grouping_policy=preset["grouping_policy"],
        fallback_policy=preset["fallback_policy"],
    )


def oracle_min_planes(
    head: "ResidualTritPlaneHead",
    hidden: torch.Tensor,
    state_context: Any,
    legal_actions: list[str],
    *,
    max_planes: int,
    accepted_action: str | None,
) -> int:
    """Offline diagnostic: smallest plane prefix that preserves ``accepted_action``.

    This is never used as a deployable method; it is only a diagnostic for
    agreement analysis.
    """
    from slm_training.models.local_action_head import ResidualTritPlaneHead

    if not isinstance(head, ResidualTritPlaneHead):
        raise TypeError("oracle_min_planes requires ResidualTritPlaneHead")
    if accepted_action is None or len(legal_actions) <= 1:
        return 0
    for p in range(0, max_planes + 1):
        out = head.score(
            hidden,
            state_context,
            legal_actions,
            max_planes=p,
        )
        action, _, _ = _decode_head(head, out, legal_actions)
        if action == accepted_action:
            return p
    return max_planes
