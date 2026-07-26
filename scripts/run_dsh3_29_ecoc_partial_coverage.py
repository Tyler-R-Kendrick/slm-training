#!/usr/bin/env python3
"""Run DSH3-29's small synthetic cost-aware ternary ECOC coverage experiment.

Builds one small synthetic four-action fixture (one uniquely catastrophic
pair, two cheap/interchangeable actions), rotates which action is the gold
target across a handful of synthetic states, builds the 100/75/50/25%
controlled-PARTIAL coverage ladder for each, and evaluates cost-aware versus
uniform :class:`TernaryECOCHead` codeword assignment plus a structural
``LocalFlatHead`` coverage baseline (no ECOC codeword geometry, so no
cost-weighted risk is computed for it -- see the ``local_flat_head_baseline``
note in the report). This is a fixture-scale wiring demonstration, never a
ship claim: see ``docs/design/dsh3-29-ecoc-partial-coverage.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.operators import (
    EffectDeltaKind,
    LegalSetCoverage,
    OperatorSupportVerdict,
)
from slm_training.models.action_code_registry import ActionCodeRegistry
from slm_training.models.local_action_head import TernaryECOCHead
from slm_training.models.operator_ecoc_coverage import (
    OperatorEcocCoverageReportV1,
    build_partial_ladder,
    evaluate_ecoc_under_coverage,
)
from slm_training.models.operator_policy_objective import (
    ControlledPartialFixtureV1,
    OperatorPolicyObjectiveV1,
    OperatorPolicyTargetV1,
    TargetUtilityLabel,
)
from slm_training.models.operator_policy_view import OperatorActionViewV1
from slm_training.models.semantic_cost import (
    OPERATOR_ACTION_COST_MATRIX_SOURCE,
    build_operator_action_cost_matrix,
)
from slm_training.versioning import build_version_stamp

HIDDEN_DIM = 8


def _action_view(
    row: int,
    operator_id: str,
    cost: float,
    effect_signature: tuple[EffectDeltaKind, ...],
    locality: str,
) -> OperatorActionViewV1:
    return OperatorActionViewV1(
        row=row,
        operator_id=operator_id,
        operator_version="v1",
        locality=locality,
        cost=cost,
        effect_signature=effect_signature,
        argument_slots=(),
        verdict=OperatorSupportVerdict.SUPPORTED,
        coverage=LegalSetCoverage.COMPLETE,
    )


# One small synthetic four-action fixture: alpha/beta are uniquely
# catastrophic to confuse with each other; delta/gamma are cheap and mutually
# interchangeable. Rotating which of the four is "gold" below (`_STATES`)
# gives a handful of distinct synthetic decision states from one fixture,
# per the issue's "keep it tight, modest first cut" scope.
_ROWS = (
    _action_view(
        0,
        "alpha_catastrophic",
        0.0,
        (EffectDeltaKind.SCOPE, EffectDeltaKind.CARDINALITY),
        "node",
    ),
    _action_view(
        1,
        "beta_catastrophic",
        0.0,
        (EffectDeltaKind.PROPERTY, EffectDeltaKind.TOPOLOGY),
        "node",
    ),
    _action_view(2, "delta_safe", 0.0, (), "node"),
    _action_view(3, "gamma_safe", 0.0, (), "node"),
)
_KEYS: tuple[str, ...] = tuple(row.operator_id for row in _ROWS)


def _objective(gold: str) -> OperatorPolicyObjectiveV1:
    targets = tuple(
        OperatorPolicyTargetV1(
            key,
            OperatorSupportVerdict.SUPPORTED,
            TargetUtilityLabel.POSITIVE if key == gold else TargetUtilityLabel.NEGATIVE,
        )
        for key in _KEYS
    )
    return OperatorPolicyObjectiveV1(coverage=LegalSetCoverage.COMPLETE, targets=targets)


def _sum_levels(reports: list[OperatorEcocCoverageReportV1]) -> list[dict[str, Any]]:
    """Sum per-level counts across states sharing one coverage-ladder shape."""
    n_levels = len(reports[0].coverage_levels)
    merged: list[dict[str, Any]] = []
    for i in range(n_levels):
        levels = [report.coverage_levels[i] for report in reports]
        total_decisions = sum(level.total_decisions for level in levels)
        cost = sum(level.cost_weighted_risk * level.total_decisions for level in levels)
        plain = sum(level.plain_risk * level.total_decisions for level in levels)
        merged.append(
            {
                "budget": levels[0].budget,
                "coverage_fraction": levels[0].coverage_fraction,
                "invalid_codewords": sum(level.invalid_codewords for level in levels),
                "defer_count": sum(level.defer_count for level in levels),
                "model_abstain_count": sum(level.model_abstain_count for level in levels),
                "compiler_defer_count": sum(level.compiler_defer_count for level in levels),
                "retained_wrong_actions": sum(level.retained_wrong_actions for level in levels),
                "detected_errors": sum(level.detected_errors for level in levels),
                "correct": sum(level.correct for level in levels),
                "total_decisions": total_decisions,
                "cost_weighted_risk": cost / total_decisions if total_decisions else 0.0,
                "plain_risk": plain / total_decisions if total_decisions else 0.0,
            }
        )
    return merged


def _local_flat_baseline(
    ladders: list[tuple[ControlledPartialFixtureV1, ...]],
) -> list[dict[str, Any]]:
    """Coverage-structure context for the non-ECOC baseline.

    ``LocalFlatHead`` has no codeword geometry, so there is no cost-weighted
    confusion risk to compute for it -- this only reports whether the gold
    action survives truncation and whether the level is a forced singleton,
    decoupled from any head family.
    """
    n_levels = len(ladders[0])
    rows: list[dict[str, Any]] = []
    for i in range(n_levels):
        fixtures = [ladder[i] for ladder in ladders]
        gold_witnessed = sum(
            any(target.utility is TargetUtilityLabel.POSITIVE for target in fixture.public.targets)
            for fixture in fixtures
        )
        forced_singleton = sum(1 for fixture in fixtures if len(fixture.public.targets) == 1)
        rows.append(
            {
                "budget": fixtures[0].budget,
                "states": len(fixtures),
                "gold_witnessed": gold_witnessed,
                "forced_singleton": forced_singleton,
            }
        )
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DSH3-29 cost-aware ternary ECOC abstention under controlled PARTIAL coverage (SLM-404)",
        "",
        "Status: fixture-scale synthetic demonstration; not a ship claim",
        "",
        f"## Aggregated per-level risk (summed over {report['run']['states']} synthetic states)",
        "",
        "| Budget | Coverage | Arm | Detected errors | Retained wrong | Cost-weighted risk | Plain risk |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for arm_name in ("cost_aware", "uniform"):
        for level in report[arm_name]["levels"]:
            lines.append(
                f"| {level['budget']} | {level['coverage_fraction']:.2f} | `{arm_name}` | "
                f"{level['detected_errors']} | {level['retained_wrong_actions']} | "
                f"{level['cost_weighted_risk']:.3f} | {level['plain_risk']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## LocalFlatHead baseline (coverage structure only)",
            "",
            report["local_flat_head_baseline"]["note"],
            "",
            "| Budget | States | Gold witnessed | Forced singleton |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for level in report["local_flat_head_baseline"]["levels"]:
        lines.append(
            f"| {level['budget']} | {level['states']} | {level['gold_witnessed']} | "
            f"{level['forced_singleton']} |"
        )
    lines.extend(
        [
            "",
            "## Honesty",
            "",
            f"Overall cost-weighted risk: cost-aware={report['cost_aware']['overall_cost_weighted_risk']:.4f}, "
            f"uniform={report['uniform']['overall_cost_weighted_risk']:.4f}. This is a hand-built, small "
            f"synthetic fixture (one catastrophic action pair rotated across {report['run']['states']} gold "
            "labels), not a real operator_policy_corpus sample -- it demonstrates the wiring end to end, not "
            "a production readiness claim.",
            "",
            f"Decision: `{report['decision']['verdict']}` -- {report['decision']['reason']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cost_matrix = build_operator_action_cost_matrix(_ROWS)
    objectives = [_objective(gold=key) for key in _KEYS]

    cost_aware_reports: list[OperatorEcocCoverageReportV1] = []
    uniform_reports: list[OperatorEcocCoverageReportV1] = []
    ladders: list[tuple[ControlledPartialFixtureV1, ...]] = []
    for objective in objectives:
        ladder = build_partial_ladder(objective, cost_matrix, seed=0)
        ladders.append(ladder)
        head_cost_aware = TernaryECOCHead(
            HIDDEN_DIM, registry=ActionCodeRegistry(), costs=cost_matrix, use_detection=True
        )
        head_uniform = TernaryECOCHead(
            HIDDEN_DIM, registry=ActionCodeRegistry(), costs=None, use_detection=True
        )
        cost_aware_reports.append(
            evaluate_ecoc_under_coverage(
                head_cost_aware,
                ladder,
                cost_matrix,
                cost_matrix_source=OPERATOR_ACTION_COST_MATRIX_SOURCE,
            )
        )
        uniform_reports.append(
            evaluate_ecoc_under_coverage(
                head_uniform, ladder, cost_matrix, cost_matrix_source="uniform"
            )
        )

    cost_aware_levels = _sum_levels(cost_aware_reports)
    uniform_levels = _sum_levels(uniform_reports)
    local_flat_levels = _local_flat_baseline(ladders)

    cost_aware_total = sum(level["total_decisions"] for level in cost_aware_levels)
    uniform_total = sum(level["total_decisions"] for level in uniform_levels)
    cost_aware_overall = (
        sum(level["cost_weighted_risk"] * level["total_decisions"] for level in cost_aware_levels)
        / cost_aware_total
        if cost_aware_total
        else 0.0
    )
    uniform_overall = (
        sum(level["cost_weighted_risk"] * level["total_decisions"] for level in uniform_levels)
        / uniform_total
        if uniform_total
        else 0.0
    )

    stamp = build_version_stamp(
        "model.operator_ecoc_coverage",
        "model.operator_policy_objective",
        "model.operator_policy_view",
    )

    if cost_aware_overall < uniform_overall:
        decision = {
            "verdict": "fixture-scale-positive-signal",
            "reason": (
                "cost-aware ECOC codeword assignment strictly reduced cost-weighted retained-wrong-action "
                "risk versus uniform assignment on this small synthetic fixture; this is fixture-scale "
                "positive signal, not a ship claim -- real-corpus PARTIAL-coverage incidence remains too low "
                "per DSH3-28 to run this at scale, so re-run once operator_policy_corpus supplies natural "
                "PARTIAL rows before treating this as more than wiring evidence"
            ),
        }
    else:
        decision = {
            "verdict": "no-benefit-observed",
            "reason": (
                "cost-aware ECOC codeword assignment did not reduce cost-weighted risk versus uniform on this "
                "fixture; per the preregistered stop rule (ECOC abstentions uncorrelated with error cost / no "
                "selective-risk benefit -> retain the simpler best DSH3-28 head), this does not overturn the "
                "standing DSH3-28 disposition"
            ),
        }

    report = {
        "schema": "dsh3_29_ecoc_partial_coverage_report/v1",
        "issue": "SLM-404",
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "states": len(objectives),
            "actions_per_state": len(_KEYS),
            "hidden_dim": HIDDEN_DIM,
            "ship_claim": False,
        },
        "cost_matrix_source": OPERATOR_ACTION_COST_MATRIX_SOURCE,
        "cost_aware": {
            "levels": cost_aware_levels,
            "overall_cost_weighted_risk": cost_aware_overall,
        },
        "uniform": {
            "levels": uniform_levels,
            "overall_cost_weighted_risk": uniform_overall,
        },
        "local_flat_head_baseline": {
            "note": (
                "LocalFlatHead has no ECOC codeword geometry, so no cost-weighted confusion risk is computed "
                "for it; this is coverage-structure context only (gold-witnessed / forced-singleton counts), "
                "not a risk comparison."
            ),
            "levels": local_flat_levels,
        },
        "decision": decision,
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(output_dir / "report.json"), "decision": decision}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
