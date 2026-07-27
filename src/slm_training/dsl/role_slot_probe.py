"""Cross-example role-swap, dropout, and truncation probes (AP-025 / SLM-318).

Pure, model-free interventions over an :class:`AbstractPlanV1` sequence of
codebook-local slot indices (``AbstractPlanTrace.plan_tokens``, one value per
round), targeted at a single declared :class:`RoleSpanV1` family at a time so
a caller can attribute a measured semantic delta to that role rather than to
the whole plan. Mirrors the orchestration shape of
:mod:`slm_training.harnesses.preference.counterfactual_probe`: the
intervention functions are pure and deterministic, and the only
model-dependent surface is the caller-supplied :class:`SemanticScorer`.

No claim of causal necessity follows from a nonzero delta alone -- that
requires the paired-example, confidence-interval evidence AP-022 (SLM-313)
already owns. ``causal_effect_matrix`` returns the raw per-(role,
intervention) effect magnitudes for that gate to consume, not a verdict.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from slm_training.dsl.abstract_plan import AbstractPlanV1

__all__ = [
    "SemanticScorer",
    "RoleProbeConfig",
    "RoleProbeResult",
    "slot_roles",
    "swap_role",
    "dropout_role",
    "truncate_role",
    "run_role_probes",
    "causal_effect_matrix",
]


class SemanticScorer(Protocol):
    """The only model-dependent surface: scores one plan-token sequence."""

    def score(self, plan_tokens: tuple[int, ...]) -> float: ...


@dataclass(frozen=True)
class RoleProbeConfig:
    # Empty means "every role declared on the plan".
    roles: tuple[str, ...] = ()
    # Rounds of a role to keep under the "truncate" intervention.
    truncate_keep: int = 1

    def __post_init__(self) -> None:
        if self.truncate_keep < 1:
            raise ValueError("truncate_keep must be >= 1")


@dataclass(frozen=True)
class RoleProbeResult:
    role: str
    intervention: str  # "swap" | "dropout" | "truncate"
    baseline_score: float
    intervened_score: float

    @property
    def delta(self) -> float:
        return self.intervened_score - self.baseline_score


def slot_roles(plan: AbstractPlanV1, plan_tokens: Sequence[int]) -> tuple[str | None, ...]:
    """The role (or None) each round's emitted slot index belongs to."""
    return tuple(plan.role_for_slot(t) for t in plan_tokens)


def _require_role_factorized(plan: AbstractPlanV1) -> None:
    if not plan.is_role_factorized:
        raise ValueError("role-slot probes require an AbstractPlanV1 with role_spans set")


def swap_role(
    plan: AbstractPlanV1,
    tokens_a: Sequence[int],
    tokens_b: Sequence[int],
    role: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Exchange the rounds belonging to ``role`` between two same-length plans.

    A round is exchanged when either sequence's emitted token at that
    position belongs to ``role`` -- every other round is left untouched in
    both sequences, so this is a targeted single-factor cross-example
    intervention, not a full-plan shuffle.
    """
    _require_role_factorized(plan)
    plan.role_span(role)  # raises KeyError if unknown, fail closed
    if len(tokens_a) != len(tokens_b):
        raise ValueError("swap_role requires equal-length plan-token sequences")
    new_a = list(tokens_a)
    new_b = list(tokens_b)
    for i, (a, b) in enumerate(zip(tokens_a, tokens_b)):
        if plan.role_for_slot(a) == role or plan.role_for_slot(b) == role:
            new_a[i], new_b[i] = b, a
    return tuple(new_a), tuple(new_b)


def dropout_role(
    plan: AbstractPlanV1, tokens: Sequence[int], role: str
) -> tuple[int, ...]:
    """Replace ``role``'s rounds with that role's first slot, a deterministic
    content-null filler analogous to ``PlanConnectorArm.EMPTY``."""
    _require_role_factorized(plan)
    span = plan.role_span(role)
    filler = span.start
    return tuple(filler if plan.role_for_slot(t) == role else t for t in tokens)


def truncate_role(
    plan: AbstractPlanV1,
    tokens: Sequence[int],
    role: str,
    *,
    keep: int | None = None,
) -> tuple[int, ...]:
    """Keep only the first ``keep`` occupied rounds of ``role``, filling the
    rest with that role's filler slot. Defaults to the span's ``max_length``
    budget (or its full length when unset)."""
    _require_role_factorized(plan)
    span = plan.role_span(role)
    limit = keep if keep is not None else (span.max_length or span.length)
    if limit < 1:
        raise ValueError("keep must be >= 1")
    filler = span.start
    result = list(tokens)
    occupied = 0
    for i, t in enumerate(tokens):
        if plan.role_for_slot(t) == role:
            occupied += 1
            if occupied > limit:
                result[i] = filler
    return tuple(result)


def run_role_probes(
    plan: AbstractPlanV1,
    pairs: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    scorer: SemanticScorer,
    *,
    config: RoleProbeConfig = RoleProbeConfig(),
) -> list[RoleProbeResult]:
    """Run swap/dropout/truncate interventions for every configured role
    against every paired example, scored by ``scorer``.

    Each ``pairs`` entry is ``(tokens_a, tokens_b)``, two same-length
    plan-token sequences (e.g. from two different training examples); the
    swap intervention consumes both, dropout/truncate only ``tokens_a``.
    """
    _require_role_factorized(plan)
    roles = config.roles or plan.role_names
    results: list[RoleProbeResult] = []
    for tokens_a, tokens_b in pairs:
        baseline = scorer.score(tuple(tokens_a))
        for role in roles:
            swapped_a, _ = swap_role(plan, tokens_a, tokens_b, role)
            results.append(
                RoleProbeResult(role, "swap", baseline, scorer.score(swapped_a))
            )
            dropped = dropout_role(plan, tokens_a, role)
            results.append(
                RoleProbeResult(role, "dropout", baseline, scorer.score(dropped))
            )
            truncated = truncate_role(plan, tokens_a, role, keep=config.truncate_keep)
            results.append(
                RoleProbeResult(role, "truncate", baseline, scorer.score(truncated))
            )
    return results


def causal_effect_matrix(
    results: Sequence[RoleProbeResult],
) -> dict[str, dict[str, float]]:
    """Mean absolute delta per (role, intervention): the raw cross-factor
    effect matrix. Not a significance test or a causal-use claim -- see the
    module docstring."""
    sums: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in results:
        key = (r.role, r.intervention)
        sums[key] += abs(r.delta)
        counts[key] += 1
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for (role, intervention), total in sums.items():
        matrix[role][intervention] = total / counts[(role, intervention)]
    return dict(matrix)
