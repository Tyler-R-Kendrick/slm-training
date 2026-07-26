"""Collapse-derived CONFLICT vs DIFFERENT_RESULT hard-negative ablation (DSH3-30).

DSH3-24/SLM-399 (`slm_training.data.flow.operator_policy_corpus`) re-projects
each collapse-derived adjacent-swap hard negative
(`slm_training.dsl.operators.collapse.CollapsedHardNegativeV1`) onto one
step's freshly re-enumerated live legal set as an
`OperatorPolicyHardNegativeV1`. That re-projection already carries a
structural signal this module makes explicit and testable: a `CONFLICT`
negative means the swapped ordering was outright illegal at that state (the
replay raised), so the alternate operator is very likely *not* among that
step's live `action_rows` (`alternate_action_row is None`) -- there is no live
alternate to score, so no ranking/margin signal is computable. A
`DIFFERENT_RESULT` negative means the swap replayed successfully but
disagreed, so the alternate operator *was* a legal, live candidate action at
that step (`alternate_action_row is not None`) -- there is a concrete
alternate-action row to rank the accepted action against.

This module does not train anything and does not touch
`slm_training.harnesses.experiments.typed_operator_policy` (DSH3-28's frozen
scorer). It operates directly on `OperatorPolicyRowV1` rows: selects which
rows contribute hard-negative supervision under one of five matched arms,
scores the accepted/alternate actions with a small deterministic
label-free proxy, and reports how densely each arm's selection actually
supports pairwise ranking -- the "usable negative rate" this issue's own
hypothesis (denser, more sample-efficient ordering supervision from
`DIFFERENT_RESULT`) is about.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from slm_training.data.flow.operator_policy_corpus import OperatorPolicyRowV1
from slm_training.dsl.operators import (
    EffectDeltaKind,
    LegalSetCoverage,
    OperatorSupportVerdict,
)
from slm_training.dsl.operators.collapse import HardNegativeOutcome
from slm_training.models.operator_policy_view import OperatorActionViewV1

__all__ = [
    "HardNegativeArm",
    "CollapseNegativeAblationReportV1",
    "HardNegativeArmResultV1",
    "select_hard_negative_rows",
    "hard_negative_margin_loss",
    "score_action_view",
    "evaluate_hard_negative_arm",
    "evaluate_hard_negative_arms",
]

#: The five matched arms DSH3-30 asks for. ``"none"`` is the baseline: no
#: hard-negative supervision contributes at all (not "rows without a hard
#: negative" -- see ``select_hard_negative_rows``).
HardNegativeArm = Literal[
    "none",
    "conflict_only",
    "different_result_only",
    "equal_mix",
    "curriculum_mix",
]

_ALL_ARMS: tuple[HardNegativeArm, ...] = (
    "none",
    "conflict_only",
    "different_result_only",
    "equal_mix",
    "curriculum_mix",
)


def _outcome_rows(
    rows: Sequence[OperatorPolicyRowV1], outcome: HardNegativeOutcome
) -> tuple[OperatorPolicyRowV1, ...]:
    return tuple(
        row
        for row in rows
        if row.hard_negative is not None and row.hard_negative.outcome is outcome
    )


def select_hard_negative_rows(
    rows: Sequence[OperatorPolicyRowV1],
    arm: HardNegativeArm,
    *,
    seed: int = 0,
) -> tuple[OperatorPolicyRowV1, ...]:
    """Select which rows contribute hard-negative training signal for ``arm``.

    - ``"none"``: empty tuple -- the baseline of no hard-negative supervision
      at all. This is a deliberate design choice, not shorthand for "rows
      whose ``hard_negative`` is ``None``".
    - ``"conflict_only"``: every row whose ``hard_negative.outcome`` is
      ``CONFLICT``.
    - ``"different_result_only"``: every row whose ``hard_negative.outcome``
      is ``DIFFERENT_RESULT``.
    - ``"equal_mix"``: balances to ``min(count_conflict,
      count_different_result)`` of each type. Deterministic: each outcome's
      rows are sorted by ``row_id``, a seeded shuffle breaks ties only within
      an identical ``row_id`` group (which never actually occurs since
      ``row_id`` is unique, so this is a no-op in practice and only present
      for symmetry with ``operator_ecoc_coverage.build_partial_ladder``'s
      tie-break convention), and the first ``N`` of each type is taken.
    - ``"curriculum_mix"``: all ``DIFFERENT_RESULT`` rows (sorted by
      ``row_id``) before all ``CONFLICT`` rows (sorted by ``row_id``). This is
      a scoped simplification of "easy negatives first" curriculum learning:
      there is no real iterative training loop here to actually stage a
      curriculum across training steps, so this only fixes a presentation
      *order* over a single static row sequence, not a real multi-phase
      curriculum schedule.

    Rows with ``hard_negative is None`` are never selected by any arm.
    """
    conflict_rows = tuple(
        sorted(_outcome_rows(rows, HardNegativeOutcome.CONFLICT), key=lambda r: r.row_id)
    )
    different_result_rows = tuple(
        sorted(
            _outcome_rows(rows, HardNegativeOutcome.DIFFERENT_RESULT),
            key=lambda r: r.row_id,
        )
    )

    if arm == "none":
        return ()
    if arm == "conflict_only":
        return conflict_rows
    if arm == "different_result_only":
        return different_result_rows
    if arm == "equal_mix":
        budget = min(len(conflict_rows), len(different_result_rows))

        def _balanced(group: tuple[OperatorPolicyRowV1, ...]) -> tuple[OperatorPolicyRowV1, ...]:
            rng = random.Random(seed)
            by_row_id: dict[str, list[OperatorPolicyRowV1]] = {}
            for row in group:
                by_row_id.setdefault(row.row_id, []).append(row)
            ordered: list[OperatorPolicyRowV1] = []
            for row_id in sorted(by_row_id):
                tie_group = by_row_id[row_id]
                rng.shuffle(tie_group)
                ordered.extend(tie_group)
            return tuple(ordered[:budget])

        return _balanced(different_result_rows) + _balanced(conflict_rows)
    if arm == "curriculum_mix":
        return different_result_rows + conflict_rows
    raise ValueError(f"unknown hard negative arm: {arm!r}")


def score_action_view(action: OperatorActionViewV1) -> float:
    """Small, deterministic, label-free proxy score for one candidate action.

    This is a fixture-scale stand-in for a trained scorer, not a real model:
    it is a fixed, deterministic function of ``OperatorActionViewV1``'s own
    sanitized fields only (``cost`` and ``effect_signature``) -- never the
    outcome/target being predicted, since reading anything about which action
    is "correct" here would leak the label into the "model". The exact
    formula is arbitrary (lower declared cost and a smaller effect signature
    score higher) and is not claimed to correlate with any real trained
    policy's behavior; it only needs to be a well-defined, side-effect-free
    function of the same inputs regardless of which arm or row it is used
    for, so the margin-loss mechanics below are exercised identically across
    arms.
    """
    return -action.cost - 0.1 * len(action.effect_signature)


def hard_negative_margin_loss(
    accepted_score: float,
    alternate_score: float | None,
    *,
    margin: float = 1.0,
) -> float | None:
    """Plain hinge/margin loss between an accepted and alternate action score.

    Returns ``None`` when ``alternate_score is None`` -- there is no live
    alternate to rank against, which is the load-bearing case for ``CONFLICT``
    rows that structurally lack a live alternate action. Otherwise returns
    ``max(0.0, margin - (accepted_score - alternate_score))``, a standard
    hinge value (``0.0`` exactly when the accepted action already clears the
    alternate by at least ``margin``).
    """
    if alternate_score is None:
        return None
    return max(0.0, margin - (accepted_score - alternate_score))


@dataclass(frozen=True)
class HardNegativeArmResultV1:
    """One arm's selection/usability breakdown over a fixed row set."""

    arm: str
    total_rows: int
    usable_rows: int
    unusable_rows: int
    negative_type_counts: dict[str, int]
    ranking_correct: int
    mean_margin_loss: float
    usable_negative_rate: float
    schema: str = "hard_negative_arm_result/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arm": self.arm,
            "total_rows": self.total_rows,
            "usable_rows": self.usable_rows,
            "unusable_rows": self.unusable_rows,
            "negative_type_counts": dict(self.negative_type_counts),
            "ranking_correct": self.ranking_correct,
            "mean_margin_loss": self.mean_margin_loss,
            "usable_negative_rate": self.usable_negative_rate,
        }


@dataclass(frozen=True)
class CollapseNegativeAblationReportV1:
    """Per-arm CONFLICT-vs-DIFFERENT_RESULT hard-negative ablation report."""

    arms: tuple[HardNegativeArmResultV1, ...]
    best_arm_by_usable_negative_rate: str
    schema: str = "collapse_negative_ablation_report/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arms": [arm.to_dict() for arm in self.arms],
            "best_arm_by_usable_negative_rate": self.best_arm_by_usable_negative_rate,
        }


def evaluate_hard_negative_arm(
    rows: Sequence[OperatorPolicyRowV1],
    arm: HardNegativeArm,
    *,
    seed: int = 0,
    margin: float = 1.0,
) -> HardNegativeArmResultV1:
    """Evaluate one arm's selection over ``rows``.

    ``usable_rows`` are selected rows whose ``hard_negative.alternate_action_row``
    is not ``None`` (a live alternate score is computable); ``unusable_rows``
    are selected rows where it is ``None`` (no ranking signal possible).
    ``usable_negative_rate = usable_rows / total_rows`` (``0.0`` when no rows
    were selected) is the central "denser supervision per example" metric
    this issue's hypothesis is about. ``ranking_correct`` counts usable rows
    where the accepted action's proxy score strictly clears the alternate's
    by at least ``margin`` (``hard_negative_margin_loss(...) == 0.0``).
    ``mean_margin_loss`` averages the margin loss over usable rows only
    (``0.0`` when there are none).
    """
    selected = select_hard_negative_rows(rows, arm, seed=seed)
    negative_type_counts: dict[str, int] = {}
    usable_rows = 0
    unusable_rows = 0
    ranking_correct = 0
    margin_losses: list[float] = []

    for row in selected:
        hard_negative = row.hard_negative
        assert hard_negative is not None  # selection never yields negative-free rows
        negative_type_counts[hard_negative.outcome.value] = (
            negative_type_counts.get(hard_negative.outcome.value, 0) + 1
        )
        action_rows = row.policy_input.get("action_rows", [])
        if hard_negative.alternate_action_row is None:
            unusable_rows += 1
            continue
        usable_rows += 1
        # ``policy_input.action_rows`` is densely row-indexed from 0 (an
        # ``OperatorPolicyInputV1`` invariant), so a plain list index matches
        # each row's own ``accepted_action_row`` / ``alternate_action_row``
        # numbering -- the same convention ``OperatorPolicyRowV1.__post_init__``
        # uses.
        accepted_action = action_rows[row.accepted_action_row]
        alternate_action = action_rows[hard_negative.alternate_action_row]
        accepted_score = score_action_view(_action_view_from_dict(accepted_action))
        alternate_score = score_action_view(_action_view_from_dict(alternate_action))
        loss = hard_negative_margin_loss(accepted_score, alternate_score, margin=margin)
        assert loss is not None
        margin_losses.append(loss)
        if loss == 0.0:
            ranking_correct += 1

    total_rows = len(selected)
    mean_margin_loss = sum(margin_losses) / len(margin_losses) if margin_losses else 0.0
    usable_negative_rate = usable_rows / total_rows if total_rows else 0.0

    return HardNegativeArmResultV1(
        arm=arm,
        total_rows=total_rows,
        usable_rows=usable_rows,
        unusable_rows=unusable_rows,
        negative_type_counts=negative_type_counts,
        ranking_correct=ranking_correct,
        mean_margin_loss=mean_margin_loss,
        usable_negative_rate=usable_negative_rate,
    )


def _action_view_from_dict(action: dict[str, Any]) -> OperatorActionViewV1:
    """Rebuild the minimal ``OperatorActionViewV1`` fields ``score_action_view``
    needs directly from a persisted ``policy_input['action_rows']`` payload
    (``OperatorActionViewV1.to_dict()`` shape), without widening what this
    module reads -- only ``cost`` and ``effect_signature`` are used."""
    return OperatorActionViewV1(
        row=action["row"],
        operator_id=action["operator_id"],
        operator_version=action["operator_version"],
        locality=action["locality"],
        cost=action["cost"],
        effect_signature=tuple(
            EffectDeltaKind(kind) for kind in action["effect_signature"]
        ),
        argument_slots=(),
        verdict=OperatorSupportVerdict(action["verdict"]),
        coverage=LegalSetCoverage(action["coverage"]),
    )


def evaluate_hard_negative_arms(
    rows: Sequence[OperatorPolicyRowV1],
    arms: Sequence[HardNegativeArm] = _ALL_ARMS,
    *,
    seed: int = 0,
    margin: float = 1.0,
) -> CollapseNegativeAblationReportV1:
    """Evaluate every arm in ``arms`` over the same fixed ``rows`` and report."""
    results = tuple(
        evaluate_hard_negative_arm(rows, arm, seed=seed, margin=margin) for arm in arms
    )
    best_arm = max(results, key=lambda result: result.usable_negative_rate).arm
    return CollapseNegativeAblationReportV1(
        arms=results, best_arm_by_usable_negative_rate=best_arm
    )
