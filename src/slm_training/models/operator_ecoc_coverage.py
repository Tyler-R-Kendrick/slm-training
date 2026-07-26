"""Cost-aware ternary ECOC abstention under controlled PARTIAL coverage (DSH3-29).

Builds a 100/75/50/25% legal-set coverage ladder from one COMPLETE
``OperatorPolicyObjectiveV1`` (reusing the DSH3-25 controlled-partial-fixture
shadow-set machinery -- this module never re-derives that truncation logic),
hides the highest-cost-of-confusion actions first at each shrinking coverage
level, and reports how a :class:`TernaryECOCHead` resolves decode ambiguity at
each level: a caught error (``detected_error``), a model-uncertainty defer
(``abstain``/``refine``), a compiler-level defer (the gold action was
truncated out of the public view entirely), a silently retained wrong action,
or a correct decode. The point of the module is to make that classification
cost-weighted, so a cost-aware codeword assignment's benefit over a uniform
one is visible as a lower cost-weighted risk, not just a lower error count.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from slm_training.dsl.operators import LegalSetCoverage
from slm_training.harnesses.distill.operator_teacher_ceiling import selective_risk_aurc
from slm_training.models.local_action_head import StateContext, TernaryECOCHead
from slm_training.models.operator_policy_objective import (
    ControlledPartialFixtureV1,
    OperatorPolicyObjectiveV1,
    TargetUtilityLabel,
    build_controlled_partial_fixture,
)
from slm_training.models.semantic_cost import CostMatrix

__all__ = [
    "OperatorEcocCoverageLevelV1",
    "OperatorEcocCoverageReportV1",
    "build_partial_ladder",
    "evaluate_ecoc_under_coverage",
    "partial_coverage_levels",
]

#: The 100/75/50/25% coverage ladder DSH3-29 asks for.
_COVERAGE_FRACTIONS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)

#: Corruption budget for the decode-ambiguity probe: the number of trits
#: flipped from the gold action's codeword toward a decoy's codeword. The
#: distance-2 ECOC guarantees every *single*-trit corruption is caught
#: (``detected_error``), so a 2-trit probe is the smallest budget that can
#: expose residual, assignment-dependent confusability between two actions.
_CORRUPTION_BUDGET = 2


def partial_coverage_levels(total: int) -> tuple[int, ...]:
    """Map the 100/75/50/25% ladder onto integer budgets for ``total`` actions.

    Monotonic non-increasing and de-duplicated (small ``total`` collapses
    adjacent fractions onto the same integer budget).
    """
    if total <= 0:
        raise ValueError("total must be positive")
    levels: list[int] = []
    for fraction in _COVERAGE_FRACTIONS:
        budget = min(total, max(1, round(total * fraction)))
        if not levels or levels[-1] != budget:
            levels.append(budget)
    return tuple(levels)


def _average_confusion_cost(key: str, keys: Sequence[str], cost_matrix: CostMatrix) -> float:
    others = [other for other in keys if other != key]
    if not others:
        return 0.0
    return sum(cost_matrix.get((key, other), 0.0) for other in others) / len(others)


def build_partial_ladder(
    objective: OperatorPolicyObjectiveV1,
    cost_matrix: CostMatrix,
    *,
    seed: int = 0,
) -> tuple[ControlledPartialFixtureV1, ...]:
    """Build the 100/75/50/25% ladder, hiding costly-to-confuse actions first.

    Each level's ``truncation_order`` keeps the *lowest* average-pairwise-cost
    actions earliest, so as the budget shrinks the highest-cost-of-confusion
    actions are the first to be hidden from the public (PARTIAL) view -- at
    low coverage only cheap, easy-to-distinguish actions remain witnessed,
    which is the low/high-cost-wrong-action stratification DSH3-29 asks for.
    Ties are broken deterministically by action key; a seeded shuffle is
    applied only *within* an exact-tie group, so the ordering is reproducible
    for a fixed ``seed`` and never depends on incidental dict/set iteration
    order.
    """
    if objective.coverage is not LegalSetCoverage.COMPLETE:
        raise ValueError("build_partial_ladder requires a COMPLETE shadow domain")
    keys = [target.action_key for target in objective.targets]
    total = len(keys)
    if total == 0:
        raise ValueError("objective must have at least one target")
    scores = {key: _average_confusion_cost(key, keys, cost_matrix) for key in keys}
    grouped: dict[float, list[str]] = {}
    for key in keys:
        grouped.setdefault(scores[key], []).append(key)
    rng = random.Random(seed)
    order: list[str] = []
    for score in sorted(grouped):
        group = sorted(grouped[score])
        rng.shuffle(group)
        order.extend(group)
    budgets = partial_coverage_levels(total)
    return tuple(
        build_controlled_partial_fixture(objective, truncation_order=tuple(order), budget=budget)
        for budget in budgets
    )


def _corrupt_toward(
    source: tuple[int, ...], target: tuple[int, ...], *, budget: int
) -> tuple[int, ...]:
    """Flip up to ``budget`` trits where ``source`` differs from ``target``.

    Positions are chosen in ascending index order, so the corruption is
    deterministic given the two codewords.
    """
    corrupted = list(source)
    flipped = 0
    for position, (source_trit, target_trit) in enumerate(zip(source, target)):
        if flipped >= budget:
            break
        if source_trit != target_trit:
            corrupted[position] = target_trit
            flipped += 1
    return tuple(corrupted)


def _shadow_fingerprint(shadow: OperatorPolicyObjectiveV1) -> str:
    """Stable hash proving which exact shadow (complete) domain was scored."""
    payload = {
        "coverage": shadow.coverage.value,
        "targets": sorted(
            (target.action_key, target.compiler_support.value, target.utility.value)
            for target in shadow.targets
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class OperatorEcocCoverageLevelV1:
    """One coverage level's decode-classification breakdown.

    ``defer_count`` is the total of ``model_abstain_count`` (the head itself
    abstained/asked to refine) and ``compiler_defer_count`` (the gold action
    was truncated out of the public legal set entirely, so no model decision
    was even possible) -- these are reported separately because conflating
    model uncertainty with compiler-level UNKNOWN/PARTIAL is exactly the
    confusion DSH3-29 asks this report to avoid.
    """

    budget: int
    coverage_fraction: float
    invalid_codewords: int
    defer_count: int
    model_abstain_count: int
    compiler_defer_count: int
    retained_wrong_actions: int
    detected_errors: int
    correct: int
    cost_weighted_risk: float
    plain_risk: float
    total_decisions: int
    schema: str = "operator_ecoc_coverage_level/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "budget": self.budget,
            "coverage_fraction": self.coverage_fraction,
            "invalid_codewords": self.invalid_codewords,
            "defer_count": self.defer_count,
            "model_abstain_count": self.model_abstain_count,
            "compiler_defer_count": self.compiler_defer_count,
            "retained_wrong_actions": self.retained_wrong_actions,
            "detected_errors": self.detected_errors,
            "correct": self.correct,
            "cost_weighted_risk": self.cost_weighted_risk,
            "plain_risk": self.plain_risk,
            "total_decisions": self.total_decisions,
        }


@dataclass(frozen=True)
class OperatorEcocCoverageReportV1:
    """Selective-risk report for one ECOC head across one coverage ladder."""

    coverage_levels: tuple[OperatorEcocCoverageLevelV1, ...]
    overall_cost_weighted_risk: float
    overall_plain_risk: float
    cost_matrix_source: str
    shadow_set_provenance: str
    selective_risk_aurc: float | None = None
    schema: str = "operator_ecoc_coverage_report/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "coverage_levels": [level.to_dict() for level in self.coverage_levels],
            "overall_cost_weighted_risk": self.overall_cost_weighted_risk,
            "overall_plain_risk": self.overall_plain_risk,
            "cost_matrix_source": self.cost_matrix_source,
            "shadow_set_provenance": self.shadow_set_provenance,
            "selective_risk_aurc": self.selective_risk_aurc,
        }


def evaluate_ecoc_under_coverage(
    head: TernaryECOCHead,
    ladder: Sequence[ControlledPartialFixtureV1],
    cost_matrix: CostMatrix,
    *,
    cost_matrix_source: str = "unspecified",
) -> OperatorEcocCoverageReportV1:
    """Score one ECOC head across a coverage ladder built by :func:`build_partial_ladder`.

    At each level, the gold action (the shadow's unique ``POSITIVE`` target)
    is decoded against the highest-cost-of-confusion decoy still present in
    the public view, using a deterministic 2-trit corruption probe of the
    gold codeword toward the decoy codeword (see ``_CORRUPTION_BUDGET``) --
    this is the smallest probe that can expose assignment-dependent
    confusability once the code's guaranteed single-trit detection is
    exhausted. Each decision is bucketed into exactly one of: compiler-level
    ``defer`` (gold was truncated out of the public view), model-uncertainty
    ``defer`` (``abstain``/``refine``), ``detected_error``, a silently
    ``retained_wrong_action``, an ``invalid_codeword`` (a nearest-codeword
    fallback guess under a non-abstain invalid-code policy), or ``correct``.
    """
    if not ladder:
        raise ValueError("ladder must contain at least one coverage level")
    shadow = ladder[0].shadow_complete
    for fixture in ladder[1:]:
        if fixture.shadow_complete != shadow:
            raise ValueError("every ladder level must share one shadow_complete domain")
    gold_keys = sorted(
        target.action_key
        for target in shadow.targets
        if target.utility is TargetUtilityLabel.POSITIVE
    )
    if not gold_keys:
        raise ValueError("shadow domain must have at least one POSITIVE target")
    gold_key = gold_keys[0]
    total_actions = len(shadow.targets)

    levels: list[OperatorEcocCoverageLevelV1] = []
    aurc_rows: list[tuple[float, bool]] = []
    total_cost = 0.0
    total_decisions_all = 0
    total_wrong_all = 0

    for fixture in ladder:
        public_keys = tuple(target.action_key for target in fixture.public.targets)
        coverage_fraction = len(public_keys) / total_actions if total_actions else 0.0
        invalid_codewords = 0
        model_abstain = 0
        compiler_defer = 0
        retained_wrong = 0
        detected_errors = 0
        correct = 0
        level_cost = 0.0

        if gold_key not in public_keys:
            # The compiler-level truncation itself removed the gold action --
            # this is a PARTIAL/no-witness case, never a model decision.
            compiler_defer = 1
        elif len(public_keys) == 1:
            # Deterministic/singleton bypass: the only legal action is gold.
            correct = 1
        else:
            decoys = [key for key in public_keys if key != gold_key]
            decoy_key = max(decoys, key=lambda key: cost_matrix.get((gold_key, key), 0.0))
            entry = head.entry_for(list(public_keys))
            gold_codeword = entry.codeword_for(gold_key)
            decoy_codeword = entry.codeword_for(decoy_key)
            if gold_codeword is None or decoy_codeword is None:
                raise ValueError("ternary ECOC entry is missing an expected assignment")
            corrupted = _corrupt_toward(
                gold_codeword, decoy_codeword, budget=_CORRUPTION_BUDGET
            )
            hidden = torch.zeros(1, head.hidden_dim)
            output = head.score(
                hidden, StateContext(state_family_id="dsh3_29_ecoc_coverage"), list(public_keys)
            )
            m = len(gold_codeword)
            trits = torch.full((1, m, 3), -100.0)
            for position, digit in enumerate(corrupted):
                trits[0, position, digit] = 100.0
            output.trits = trits
            decision = head.decode(output, list(public_keys))
            if decision.decision_kind == "forced":
                if decision.action_identity == gold_key:
                    correct = 1
                else:
                    retained_wrong = 1
                    level_cost += cost_matrix.get(
                        (gold_key, decision.action_identity or ""), 0.0
                    )
            elif decision.decision_kind == "detected_error":
                detected_errors = 1
            elif decision.decision_kind in ("abstain", "refine"):
                model_abstain = 1
            elif decision.decision_kind == "scored":
                if decision.action_identity == gold_key:
                    correct = 1
                elif decision.telemetry.get("fallback") == "nearest_codeword":
                    invalid_codewords = 1
                    if decision.action_identity is not None:
                        level_cost += cost_matrix.get(
                            (gold_key, decision.action_identity), 0.0
                        )
                else:
                    retained_wrong = 1
                    if decision.action_identity is not None:
                        level_cost += cost_matrix.get(
                            (gold_key, decision.action_identity), 0.0
                        )
            else:  # pragma: no cover - decision_kind is a closed set today.
                raise ValueError(f"unexpected decision_kind {decision.decision_kind!r}")

        total_decisions = 1
        defer_count = model_abstain + compiler_defer
        plain_risk = retained_wrong / total_decisions
        cost_weighted_risk = level_cost / total_decisions
        levels.append(
            OperatorEcocCoverageLevelV1(
                budget=fixture.budget,
                coverage_fraction=coverage_fraction,
                invalid_codewords=invalid_codewords,
                defer_count=defer_count,
                model_abstain_count=model_abstain,
                compiler_defer_count=compiler_defer,
                retained_wrong_actions=retained_wrong,
                detected_errors=detected_errors,
                correct=correct,
                cost_weighted_risk=cost_weighted_risk,
                plain_risk=plain_risk,
                total_decisions=total_decisions,
            )
        )
        total_cost += level_cost
        total_decisions_all += total_decisions
        total_wrong_all += retained_wrong
        aurc_rows.append((coverage_fraction, bool(correct)))

    overall_cost_weighted_risk = (
        total_cost / total_decisions_all if total_decisions_all else 0.0
    )
    overall_plain_risk = (
        total_wrong_all / total_decisions_all if total_decisions_all else 0.0
    )
    return OperatorEcocCoverageReportV1(
        coverage_levels=tuple(levels),
        overall_cost_weighted_risk=overall_cost_weighted_risk,
        overall_plain_risk=overall_plain_risk,
        cost_matrix_source=cost_matrix_source,
        shadow_set_provenance=_shadow_fingerprint(shadow),
        selective_risk_aurc=selective_risk_aurc(aurc_rows),
    )
