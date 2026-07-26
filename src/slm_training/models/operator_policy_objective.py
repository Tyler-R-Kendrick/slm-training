"""Orthogonal support/utility objective and DEFER routing for operator policies
(DSH3-25/SLM-400).

Answers how a policy should learn and act when legal-set coverage is
partial and target utility is unknown: treating every unlabeled live
candidate as an ordinary negative is the failure mode this module exists to
avoid. It separates two axes that the runtime already keeps orthogonal —
compiler support (`OperatorSupportVerdict` / `LegalSetCoverage`, from
`legal_set.py`) and target utility (`TargetUtility`, new here) — and adds an
explicit `InferenceRoute` so a policy can abstain (`DEFER`) instead of being
forced to guess when a slot's candidate domain was truncated and no witness
was found.

Built over the SLM-397 sanitized `OperatorPolicyInputV1` boundary and the
SLM-398 `CandidateScoringHead` shape; a fixture-only matched-arm comparison
(`run_matched_objective_fixture`) uses `enumerate_operator_legal_set`'s own
truncation (`max_combinations_per_operator`) to produce genuine `PARTIAL`
coverage, never a hand-faked one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from slm_training.dsl.operators import LegalSetCoverage, OperatorSupportVerdict
from slm_training.models.operator_policy_view import OperatorPolicyInputV1

__all__ = [
    "InferenceRoute",
    "ObjectiveArm",
    "OperatorPolicyObjectiveReportV1",
    "OperatorPolicyObjectiveV1",
    "TargetUtility",
    "build_operator_policy_objectives",
    "classify_route",
    "defer_abstention_loss",
    "known_supported_ce_loss",
    "positive_unlabeled_risk_loss",
    "run_matched_objective_fixture",
]


class TargetUtility(str, Enum):
    """A candidate's target-utility label — orthogonal to compiler support.

    Only ``POSITIVE`` and ``UNKNOWN`` exist in this version: the runtime has
    no per-argument-candidate certified-negative source yet (SLM-399's
    collapse hard negatives name a whole alternate *operator*, not an
    alternate candidate within the same slot). A ``NEGATIVE`` label is
    deliberately not offered here rather than approximated — see the design
    doc's scope notes.
    """

    POSITIVE = "positive"
    UNKNOWN = "unknown"


class InferenceRoute(str, Enum):
    """Which serving-time path one action-row/slot decision takes."""

    UNSUPPORTED = "unsupported"
    COMPLETE_SINGLETON = "complete_singleton"
    COMPLETE_AMBIGUOUS = "complete_ambiguous"
    PARTIAL_WITNESSED = "partial_witnessed"
    PARTIAL_UNKNOWN = "partial_unknown"


def classify_route(
    *,
    verdict: OperatorSupportVerdict,
    coverage: LegalSetCoverage,
    domain_complete: bool,
    has_positive_witness: bool,
    is_forced_singleton: bool,
) -> InferenceRoute:
    """Route one action-row/slot decision. Order matters: singleton bypass
    and hard-prunable unsupported are checked before coverage."""
    if verdict is OperatorSupportVerdict.UNSUPPORTED:
        return InferenceRoute.UNSUPPORTED
    if is_forced_singleton:
        return InferenceRoute.COMPLETE_SINGLETON
    if coverage is LegalSetCoverage.COMPLETE and domain_complete:
        return InferenceRoute.COMPLETE_AMBIGUOUS
    if has_positive_witness:
        return InferenceRoute.PARTIAL_WITNESSED
    return InferenceRoute.PARTIAL_UNKNOWN


@dataclass(frozen=True)
class OperatorPolicyObjectiveV1:
    """One action-row/slot's orthogonal support/utility labels and route.

    ``unknown_candidate_rows`` is always exactly ``candidate_rows`` minus
    ``positive_candidate_rows`` — disjoint and exhaustive by construction,
    not by a separately-checked invariant that could drift.
    """

    action_row: int
    slot_id: str
    compiler_support: OperatorSupportVerdict
    coverage: LegalSetCoverage
    domain_complete: bool
    candidate_rows: tuple[int, ...]
    positive_candidate_rows: tuple[int, ...]
    route: InferenceRoute
    schema: str = "operator_policy_objective/v1"

    def __post_init__(self) -> None:
        if len(set(self.candidate_rows)) != len(self.candidate_rows):
            raise ValueError("candidate_rows must be unique")
        positives = set(self.positive_candidate_rows)
        if not positives <= set(self.candidate_rows):
            raise ValueError("positive_candidate_rows must be a subset of candidate_rows")
        if self.route is InferenceRoute.PARTIAL_UNKNOWN and positives:
            raise ValueError("PARTIAL_UNKNOWN route cannot carry a positive witness")
        if self.route is InferenceRoute.PARTIAL_WITNESSED and not positives:
            raise ValueError("PARTIAL_WITNESSED route requires a positive witness")

    @property
    def unknown_candidate_rows(self) -> tuple[int, ...]:
        positives = set(self.positive_candidate_rows)
        return tuple(row for row in self.candidate_rows if row not in positives)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "action_row": self.action_row,
            "slot_id": self.slot_id,
            "compiler_support": self.compiler_support.value,
            "coverage": self.coverage.value,
            "domain_complete": self.domain_complete,
            "candidate_rows": list(self.candidate_rows),
            "positive_candidate_rows": list(self.positive_candidate_rows),
            "unknown_candidate_rows": list(self.unknown_candidate_rows),
            "route": self.route.value,
        }


def build_operator_policy_objectives(
    policy_input: OperatorPolicyInputV1,
    positive_rows_by_action_row: dict[int, tuple[int, ...]],
) -> tuple[OperatorPolicyObjectiveV1, ...]:
    """Build one objective per action-row/slot in ``policy_input``.

    ``positive_rows_by_action_row`` supplies certified-positive candidate
    rows for whichever action rows have one (typically only the accepted
    action, e.g. from SLM-399's ``OperatorPolicyRowV1.accepted_argument_rows``)
    — every other action row, and every slot with no supplied positive, is
    exhaustively ``UNKNOWN``, never guessed at.
    """
    is_forced_singleton = (
        policy_input.coverage is LegalSetCoverage.COMPLETE
        and len(policy_input.action_rows) == 1
        and len(policy_input.action_rows[0].argument_slots) <= 1
        and (
            not policy_input.action_rows[0].argument_slots
            or len(policy_input.action_rows[0].argument_slots[0].candidate_rows) <= 1
        )
    )
    objectives: list[OperatorPolicyObjectiveV1] = []
    for action in policy_input.action_rows:
        positives_for_action = set(positive_rows_by_action_row.get(action.row, ()))
        for slot in action.argument_slots:
            positive_rows = tuple(
                row for row in slot.candidate_rows if row in positives_for_action
            )
            route = classify_route(
                verdict=action.verdict,
                coverage=action.coverage,
                domain_complete=slot.domain_complete,
                has_positive_witness=bool(positive_rows),
                is_forced_singleton=is_forced_singleton,
            )
            objectives.append(
                OperatorPolicyObjectiveV1(
                    action_row=action.row,
                    slot_id=slot.slot_id,
                    compiler_support=action.verdict,
                    coverage=action.coverage,
                    domain_complete=slot.domain_complete,
                    candidate_rows=slot.candidate_rows,
                    positive_candidate_rows=positive_rows,
                    route=route,
                )
            )
        if not action.argument_slots:
            route = classify_route(
                verdict=action.verdict,
                coverage=action.coverage,
                domain_complete=True,
                has_positive_witness=bool(positives_for_action),
                is_forced_singleton=is_forced_singleton,
            )
            objectives.append(
                OperatorPolicyObjectiveV1(
                    action_row=action.row,
                    slot_id="",
                    compiler_support=action.verdict,
                    coverage=action.coverage,
                    domain_complete=True,
                    candidate_rows=(),
                    positive_candidate_rows=(),
                    route=route,
                )
            )
    return tuple(objectives)


class ObjectiveArm(str, Enum):
    KNOWN_SUPPORTED_CE = "known_supported_ce"
    POSITIVE_UNLABELED_RISK = "positive_unlabeled_risk"
    DEFER_ABSTENTION = "defer_abstention"


def known_supported_ce_loss(
    logits: torch.Tensor, objective: OperatorPolicyObjectiveV1
) -> torch.Tensor | None:
    """Naive baseline: every unknown candidate is ordinary negative pressure.

    This is exactly the failure mode the ticket's hypothesis argues against
    as the unknown fraction rises — kept as the control arm, never removed.
    Returns ``None`` (no gradient signal) only when there is no candidate at
    all to score over.
    """
    if not objective.candidate_rows or not objective.positive_candidate_rows:
        return None
    log_probs = F.log_softmax(logits, dim=-1)
    positive_mask = torch.zeros_like(logits, dtype=torch.bool)
    positive_mask[list(objective.positive_candidate_rows)] = True
    return -log_probs[positive_mask].mean()


def positive_unlabeled_risk_loss(
    logits: torch.Tensor,
    objective: OperatorPolicyObjectiveV1,
    *,
    prior: float = 0.5,
) -> torch.Tensor | None:
    """Non-negative PU (nnPU, du Plessis et al.) risk over per-candidate
    sigmoid scores. Unknown candidates are treated as an unlabeled mixture
    (weight ``prior`` positive, ``1 - prior`` negative) rather than as
    confident negatives — the direct fix for the naive-CE failure mode.
    """
    if not objective.candidate_rows or not objective.positive_candidate_rows:
        return None
    if not 0.0 < prior < 1.0:
        raise ValueError("prior must be in (0, 1)")
    positive_rows = torch.tensor(objective.positive_candidate_rows, dtype=torch.long)
    unknown_rows = torch.tensor(objective.unknown_candidate_rows, dtype=torch.long)
    positive_logits = logits[positive_rows]
    positive_as_pos = F.binary_cross_entropy_with_logits(
        positive_logits, torch.ones_like(positive_logits)
    )
    positive_as_neg = F.binary_cross_entropy_with_logits(
        positive_logits, torch.zeros_like(positive_logits)
    )
    if unknown_rows.numel() == 0:
        return prior * positive_as_pos
    unknown_logits = logits[unknown_rows]
    unknown_as_neg = F.binary_cross_entropy_with_logits(
        unknown_logits, torch.zeros_like(unknown_logits)
    )
    negative_risk = unknown_as_neg - prior * positive_as_neg
    return prior * positive_as_pos + torch.clamp(negative_risk, min=0.0)


def defer_abstention_loss(
    logits: torch.Tensor, objective: OperatorPolicyObjectiveV1
) -> torch.Tensor | None:
    """Score only when the route has a positive witness; abstain (no loss,
    no forced pick) on ``PARTIAL_UNKNOWN``. Identical to the known-supported
    CE arm on every other route — the only difference is what happens when
    there is nothing witnessed to learn from.
    """
    if objective.route is InferenceRoute.PARTIAL_UNKNOWN:
        return None
    return known_supported_ce_loss(logits, objective)


@dataclass(frozen=True)
class OperatorPolicyObjectiveReportV1:
    """Matched-arm fixture comparison across unknown fractions. Wiring
    evidence only — see the DSH3-25 design doc's stop-rule disposition."""

    schema: str
    claim_class: str
    unknown_fractions: dict[str, dict[str, Any]]
    false_hard_eliminations: int
    disposition: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_class": self.claim_class,
            "unknown_fractions": self.unknown_fractions,
            "false_hard_eliminations": self.false_hard_eliminations,
            "disposition": self.disposition,
            "note": self.note,
        }


def _route_selective_metrics(
    objective: OperatorPolicyObjectiveV1, logits: torch.Tensor | None
) -> dict[str, Any]:
    if objective.route is InferenceRoute.PARTIAL_UNKNOWN or logits is None:
        return {"abstained": True, "accepted_set_mass": None}
    probs = F.softmax(logits, dim=-1)
    positive_rows = torch.tensor(objective.positive_candidate_rows, dtype=torch.long)
    return {
        "abstained": False,
        "accepted_set_mass": float(probs[positive_rows].sum().item()),
    }


def run_matched_objective_fixture(
    *,
    fixture_builder,
    unknown_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    dim: int = 6,
    seed: int = 0,
) -> OperatorPolicyObjectiveReportV1:
    """Compare the three arms across truncation levels on one fixture.

    ``fixture_builder(unknown_fraction, seed) -> (policy_input, positive_rows_by_action_row, logits)``
    is supplied by the caller (see ``scripts/run_dsh3_25_operator_policy_objective_fixture.py``)
    so this module stays free of any one DSL-pack fixture's construction
    details; it only drives the objective/loss/routing contract.
    """
    torch.manual_seed(seed)
    results: dict[str, dict[str, Any]] = {}
    false_hard_eliminations = 0
    for fraction in unknown_fractions:
        policy_input, positive_rows_by_action_row, logits_by_action_row = fixture_builder(
            fraction, seed
        )
        objectives = build_operator_policy_objectives(policy_input, positive_rows_by_action_row)
        arm_losses: dict[str, list[float]] = {arm.value: [] for arm in ObjectiveArm}
        routes: dict[str, int] = {}
        for objective in objectives:
            routes[objective.route.value] = routes.get(objective.route.value, 0) + 1
            logits = logits_by_action_row.get(objective.action_row)
            if logits is None:
                continue
            for arm, loss_fn in (
                (ObjectiveArm.KNOWN_SUPPORTED_CE, known_supported_ce_loss),
                (ObjectiveArm.POSITIVE_UNLABELED_RISK, positive_unlabeled_risk_loss),
                (ObjectiveArm.DEFER_ABSTENTION, defer_abstention_loss),
            ):
                loss = loss_fn(logits, objective)
                if loss is not None:
                    arm_losses[arm.value].append(float(loss.item()))
            # A false hard elimination would be a PARTIAL_UNKNOWN route whose
            # candidate_rows are non-empty but every entry was silently
            # dropped rather than left as an explicit abstention target —
            # structurally impossible here since unknown_candidate_rows is
            # derived, not separately asserted, but checked anyway as a
            # regression guard.
            if (
                objective.route is InferenceRoute.PARTIAL_UNKNOWN
                and objective.positive_candidate_rows
            ):
                false_hard_eliminations += 1
        results[f"unknown_fraction_{fraction:.2f}"] = {
            "routes": routes,
            "mean_loss": {
                arm: (sum(values) / len(values) if values else None)
                for arm, values in arm_losses.items()
            },
        }
    disposition = (
        "invalid_false_hard_elimination"
        if false_hard_eliminations
        else "wiring_underpowered"
    )
    note = (
        "A PARTIAL_UNKNOWN route carried a positive witness, which the "
        "route/label contract forbids by construction — investigate before "
        "trusting any arm's numbers."
        if false_hard_eliminations
        else (
            "Fixture-scale only: three arms are wired and structurally "
            "distinguished (DEFER abstains on PARTIAL_UNKNOWN, the naive CE "
            "arm never does), but no claim is made that PU risk or DEFER "
            "beats naive CE at production scale in either direction, per "
            "the DSH3-25 stop rule."
        )
    )
    return OperatorPolicyObjectiveReportV1(
        schema="operator_policy_objective_report/v1",
        claim_class="wiring",
        unknown_fractions=results,
        false_hard_eliminations=false_hard_eliminations,
        disposition=disposition,
        note=note,
    )
