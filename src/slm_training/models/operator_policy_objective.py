"""Coverage-aware supervision for the default-off DSH3 operator policy.

This module deliberately owns labels and loss accounting only.  It neither
enumerates legal actions nor changes decoder authority: compiler coverage and
the exact legal-set singleton rule remain in ``dsl.operators.legal_set``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from slm_training.dsl.operators.legal_set import (
    LegalSetCoverage,
    OperatorSupportVerdict,
)

__all__ = [
    "ControlledPartialFixtureV1",
    "DeferFallbackAttributionV1",
    "ObjectiveLossKind",
    "OperatorPolicyObjectiveV1",
    "OperatorPolicyRouteV1",
    "OperatorPolicyTargetV1",
    "TargetUtilityLabel",
    "build_controlled_partial_fixture",
    "false_hard_eliminations",
    "operator_policy_loss",
    "operator_policy_report",
    "route_operator_policy_inference",
]


class TargetUtilityLabel(str, Enum):
    """Target utility is intentionally independent of compiler support."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class ObjectiveLossKind(str, Enum):
    KNOWN_SUPPORTED_CE = "known_supported_ce"
    PU_RISK = "pu_risk"
    DEFER = "defer"


class OperatorPolicyRouteV1(str, Enum):
    COMPLETE_SINGLETON = "complete_singleton"
    COMPLETE_AMBIGUOUS = "complete_ambiguous"
    COMPLETE_EMPTY = "complete_empty"
    PARTIAL_WITNESSED = "partial_witnessed"
    PARTIAL_UNKNOWN = "partial_unknown"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class OperatorPolicyTargetV1:
    """One evaluator-only target; ``action_key`` never enters model input."""

    action_key: str
    compiler_support: OperatorSupportVerdict
    utility: TargetUtilityLabel
    schema: str = "operator_policy_target/v1"

    def __post_init__(self) -> None:
        if not self.action_key:
            raise ValueError("action_key is required")
        if (
            self.compiler_support is OperatorSupportVerdict.UNKNOWN
            and self.utility is TargetUtilityLabel.NEGATIVE
        ):
            raise ValueError("UNKNOWN compiler support cannot be an ordinary negative")
        if (
            self.compiler_support is OperatorSupportVerdict.UNSUPPORTED
            and self.utility is TargetUtilityLabel.POSITIVE
        ):
            raise ValueError("unsupported action cannot be a positive target")


@dataclass(frozen=True)
class OperatorPolicyObjectiveV1:
    """Coverage-aware labels for one policy decision.

    The public target list is the only candidate surface a learned policy may
    receive.  Its labels are supervision/evaluation metadata, not model
    features; :meth:`model_input` intentionally excludes utility and keys.
    """

    coverage: LegalSetCoverage
    targets: tuple[OperatorPolicyTargetV1, ...]
    schema: str = "operator_policy_objective/v1"

    def __post_init__(self) -> None:
        if len({target.action_key for target in self.targets}) != len(self.targets):
            raise ValueError("target action keys must be unique")
        if self.coverage is LegalSetCoverage.COMPLETE and any(
            target.compiler_support is OperatorSupportVerdict.UNKNOWN
            for target in self.targets
        ):
            raise ValueError("complete coverage cannot retain UNKNOWN support")

    @property
    def known_supported_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, target in enumerate(self.targets)
            if target.compiler_support is OperatorSupportVerdict.SUPPORTED
        )

    @property
    def known_utility_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, target in enumerate(self.targets)
            if target.compiler_support is OperatorSupportVerdict.SUPPORTED
            and target.utility is not TargetUtilityLabel.UNKNOWN
        )

    @property
    def positive_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, target in enumerate(self.targets)
            if target.compiler_support is OperatorSupportVerdict.SUPPORTED
            and target.utility is TargetUtilityLabel.POSITIVE
        )

    @property
    def unknown_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, target in enumerate(self.targets)
            if target.compiler_support is OperatorSupportVerdict.UNKNOWN
        )

    @property
    def requires_defer(self) -> bool:
        return self.coverage is LegalSetCoverage.PARTIAL or bool(self.unknown_indices)

    def model_input(self) -> dict[str, object]:
        """Return only public compiler-state metadata for a model boundary."""
        return {
            "schema": "operator_policy_objective_model_input/v1",
            "coverage": self.coverage.value,
            "candidate_support": [
                target.compiler_support.value for target in self.targets
            ],
        }


@dataclass(frozen=True)
class ControlledPartialFixtureV1:
    """A public bounded projection plus evaluator-only complete shadow truth."""

    public: OperatorPolicyObjectiveV1
    shadow_complete: OperatorPolicyObjectiveV1
    truncation_order: tuple[str, ...]
    budget: int
    schema: str = "controlled_partial_operator_policy_fixture/v1"

    def __post_init__(self) -> None:
        if self.public.coverage is not LegalSetCoverage.PARTIAL:
            raise ValueError("controlled fixture public view must be PARTIAL")
        if self.shadow_complete.coverage is not LegalSetCoverage.COMPLETE:
            raise ValueError("controlled fixture shadow must be COMPLETE")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        shadow_keys = {target.action_key for target in self.shadow_complete.targets}
        if set(self.truncation_order) != shadow_keys:
            raise ValueError("truncation order must permute complete shadow keys")
        if self.budget > len(self.truncation_order):
            raise ValueError("budget exceeds complete shadow")

    def model_input(self) -> dict[str, object]:
        """Expose the public projection only; shadow truth is evaluator-only."""
        return self.public.model_input()


@dataclass(frozen=True)
class DeferFallbackAttributionV1:
    route: OperatorPolicyRouteV1
    fallback_attempted: bool
    fallback_exact_success: bool
    schema: str = "operator_policy_defer_fallback_attribution/v1"

    def __post_init__(self) -> None:
        if self.fallback_exact_success and not self.fallback_attempted:
            raise ValueError("exact success requires a fallback attempt")

    @property
    def credited_defer(self) -> bool:
        return (
            self.route
            in {
                OperatorPolicyRouteV1.PARTIAL_WITNESSED,
                OperatorPolicyRouteV1.PARTIAL_UNKNOWN,
                OperatorPolicyRouteV1.TIMEOUT,
                OperatorPolicyRouteV1.BUDGET_EXHAUSTED,
            }
            and self.fallback_exact_success
        )


def build_controlled_partial_fixture(
    complete: OperatorPolicyObjectiveV1,
    *,
    truncation_order: Sequence[str],
    budget: int,
) -> ControlledPartialFixtureV1:
    """Project a known complete domain without leaking its hidden remainder."""
    if complete.coverage is not LegalSetCoverage.COMPLETE:
        raise ValueError("controlled partial fixtures require a complete shadow domain")
    order = tuple(truncation_order)
    by_key = {target.action_key: target for target in complete.targets}
    if set(order) != set(by_key) or len(order) != len(by_key):
        raise ValueError("truncation_order must be a permutation of complete keys")
    if not 0 <= budget <= len(order):
        raise ValueError("budget must lie inside the complete domain")
    return ControlledPartialFixtureV1(
        public=OperatorPolicyObjectiveV1(
            coverage=LegalSetCoverage.PARTIAL,
            targets=tuple(by_key[key] for key in order[:budget]),
        ),
        shadow_complete=complete,
        truncation_order=order,
        budget=budget,
    )


def route_operator_policy_inference(
    objective: OperatorPolicyObjectiveV1,
    *,
    timed_out: bool = False,
    budget_exhausted: bool = False,
) -> OperatorPolicyRouteV1:
    """Route uncertain domains to exact fallback; partial sets never force."""
    if timed_out:
        return OperatorPolicyRouteV1.TIMEOUT
    if budget_exhausted:
        return OperatorPolicyRouteV1.BUDGET_EXHAUSTED
    if objective.coverage is LegalSetCoverage.PARTIAL:
        return (
            OperatorPolicyRouteV1.PARTIAL_WITNESSED
            if objective.known_supported_indices
            else OperatorPolicyRouteV1.PARTIAL_UNKNOWN
        )
    if not objective.known_supported_indices:
        return OperatorPolicyRouteV1.COMPLETE_EMPTY
    if len(objective.known_supported_indices) == 1:
        return OperatorPolicyRouteV1.COMPLETE_SINGLETON
    return OperatorPolicyRouteV1.COMPLETE_AMBIGUOUS


def _indices_tensor(indices: Iterable[int], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(tuple(indices), dtype=torch.long, device=device)


def operator_policy_loss(
    logits: torch.Tensor,
    objective: OperatorPolicyObjectiveV1,
    *,
    kind: ObjectiveLossKind,
    defer_logit: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one safe objective; UNKNOWN is outside every negative denominator."""
    if logits.ndim != 1 or logits.numel() != len(objective.targets):
        raise ValueError("logits must have one score per public target")
    known = _indices_tensor(objective.known_supported_indices, device=logits.device)
    positives = _indices_tensor(objective.positive_indices, device=logits.device)
    labelled = _indices_tensor(objective.known_utility_indices, device=logits.device)
    zero = logits.sum() * 0.0
    denominator = 0
    if kind is ObjectiveLossKind.KNOWN_SUPPORTED_CE:
        denominator = int(known.numel())
        if not len(positives) or not len(known):
            loss = zero
        else:
            loss = torch.logsumexp(logits[known], 0) - torch.logsumexp(
                logits[positives], 0
            )
    elif kind is ObjectiveLossKind.PU_RISK:
        # Only certified utility labels participate; unlabeled/UNKNOWN targets
        # are excluded rather than being silently relabelled negative.
        denominator = int(labelled.numel())
        if not len(positives) or not len(labelled):
            loss = zero
        else:
            loss = torch.logsumexp(logits[labelled], 0) - torch.logsumexp(
                logits[positives], 0
            )
    elif kind is ObjectiveLossKind.DEFER:
        if defer_logit is None or defer_logit.numel() != 1:
            raise ValueError("DEFER objective requires one defer_logit")
        if objective.requires_defer:
            denominator = 1
            loss = F.softplus(-defer_logit.reshape(()))
        elif len(positives) and len(known):
            denominator = int(known.numel()) + 1
            all_scores = torch.cat((logits[known], defer_logit.reshape(1)))
            loss = torch.logsumexp(all_scores, 0) - torch.logsumexp(
                logits[positives], 0
            )
        else:
            loss = zero
    else:  # pragma: no cover - Enum makes this defensive only.
        raise ValueError(f"unknown objective kind: {kind}")
    return loss, {
        "denominator": float(denominator),
        "known_supported": float(len(known)),
        "unknown": float(len(objective.unknown_indices)),
        "positive": float(len(positives)),
        "coverage_complete": float(objective.coverage is LegalSetCoverage.COMPLETE),
    }


def false_hard_eliminations(fixture: ControlledPartialFixtureV1) -> int:
    """Count shadow positives incorrectly hard-pruned by the public view."""
    public_by_key = {target.action_key: target for target in fixture.public.targets}
    return sum(
        target.utility is TargetUtilityLabel.POSITIVE
        and public_by_key.get(target.action_key) is not None
        and public_by_key[target.action_key].compiler_support
        is OperatorSupportVerdict.UNSUPPORTED
        for target in fixture.shadow_complete.targets
    )


def operator_policy_report(
    scored: Sequence[tuple[OperatorPolicyObjectiveV1, torch.Tensor]],
    *,
    defer_attributions: Sequence[DeferFallbackAttributionV1] = (),
    partial_fixtures: Sequence[ControlledPartialFixtureV1] = (),
) -> dict[str, float]:
    """Compact selection report; quality/calibration are never inferred from loss."""
    accepted_mass: list[float] = []
    unknown_mass: list[float] = []
    coverage = {"complete": 0, "partial": 0}
    for objective, logits in scored:
        if logits.ndim != 1 or logits.numel() != len(objective.targets):
            raise ValueError("scored logits must align with public targets")
        coverage[objective.coverage.value] += 1
        probabilities = torch.softmax(logits, dim=0) if len(logits) else logits
        positives = _indices_tensor(objective.positive_indices, device=logits.device)
        unknown = _indices_tensor(objective.unknown_indices, device=logits.device)
        accepted_mass.append(
            float(probabilities[positives].sum()) if len(positives) else 0.0
        )
        unknown_mass.append(
            float(probabilities[unknown].sum()) if len(unknown) else 0.0
        )
    # Binary ECE for the mass assigned to the UNKNOWN set; one compact bin is
    # intentionally honest for a fixture-scale report, not a calibration claim.
    observed_unknown = [
        float(bool(objective.unknown_indices)) for objective, _ in scored
    ]
    ece = (
        abs(
            sum(unknown_mass) / len(unknown_mass)
            - sum(observed_unknown) / len(observed_unknown)
        )
        if unknown_mass
        else 0.0
    )
    return {
        "rows": float(len(scored)),
        "coverage_complete": float(coverage["complete"]),
        "coverage_partial": float(coverage["partial"]),
        "accepted_set_mass": sum(accepted_mass) / len(accepted_mass)
        if accepted_mass
        else 0.0,
        "risk": 1.0 - (sum(accepted_mass) / len(accepted_mass))
        if accepted_mass
        else 0.0,
        "unknown_mass_ece": ece,
        "false_hard_eliminations": float(
            sum(false_hard_eliminations(item) for item in partial_fixtures)
        ),
        "credited_defers": float(
            sum(item.credited_defer for item in defer_attributions)
        ),
        "uncredited_defers": float(
            sum(not item.credited_defer for item in defer_attributions)
        ),
        "partial_singleton_bypasses": float(
            sum(
                route_operator_policy_inference(item.public)
                is OperatorPolicyRouteV1.COMPLETE_SINGLETON
                for item in partial_fixtures
            )
        ),
    }
