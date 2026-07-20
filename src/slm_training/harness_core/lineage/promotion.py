"""Promotion, base-bakeoff, and human deployment gates."""

from __future__ import annotations

import math
from typing import Iterable

from slm_training.harness_core.lineage.records import EvaluationReport

MAX_ARTIFACT_BYTES = 1_000_000_000
MAX_WARM_P95_SECONDS = 15.0
NLL_REGRESSION = 0.02
METRIC_REGRESSION_POINTS = 0.02
HARD_NLL_CATEGORIES = ("binding", "structural", "repair")
HARD_METRICS = (
    "parse_rate",
    "meaningful_program_rate",
    "placeholder_fidelity",
    # Coverage of the requested slot contract. The evaluator emits this as
    # contract_recall; the old "request_coverage" name had no writer anywhere,
    # which made this gate impossible to satisfy.
    "contract_recall",
    "structural_similarity",
)


def promotion_failures(
    candidate: EvaluationReport,
    parent: EvaluationReport,
    finalist_reports: Iterable[EvaluationReport],
) -> list[str]:
    reports = list(finalist_reports)
    failures: list[str] = []
    if not candidate.ship_gates_pass or any(
        not report.ship_gates_pass for report in reports
    ):
        failures.append("every honest ship gate must pass")
    if candidate.weighted_nll is None or parent.weighted_nll is None:
        failures.append("weighted NLL is required for candidate and parent")
    elif candidate.weighted_nll >= parent.weighted_nll:
        failures.append("weighted NLL did not improve over parent")
    for category in HARD_NLL_CATEGORIES:
        if (
            category not in parent.category_nll
            or category not in candidate.category_nll
        ):
            failures.append(f"{category} NLL is required for candidate and parent")
            continue
        maximum = parent.category_nll[category] * (1 + NLL_REGRESSION)
        if candidate.category_nll[category] > maximum:
            failures.append(f"{category} NLL regressed by more than 2%")
    for metric in HARD_METRICS:
        if metric not in parent.metrics or metric not in candidate.metrics:
            failures.append(f"{metric} is required for candidate and parent")
            continue
        if (
            candidate.metrics[metric]
            < parent.metrics[metric] - METRIC_REGRESSION_POINTS
        ):
            failures.append(f"{metric} regressed by more than 2 percentage points")
    all_reports = [candidate, *reports]
    if any(
        report.metadata.get("loss_suite_complete") is not True for report in all_reports
    ):
        failures.append("loss suite is incomplete")
    if any(
        int(report.suite_sizes.get("rico_held", 0)) < 1500 for report in all_reports
    ):
        failures.append(
            "production promotion requires rico_held n>=1500 for every finalist"
        )
    if len({report.seed for report in all_reports}) < 3:
        failures.append("finalist requires three distinct seeds")
    if not {1.0, 3.0}.issubset({report.token_rung for report in all_reports}):
        failures.append("finalist requires the two largest token rungs (1x and 3x)")
    if any(report.metadata.get("ranking_stable") is not True for report in all_reports):
        failures.append("finalist ranking is not stable")
    if (
        candidate.artifact_size_bytes is None
        or candidate.artifact_size_bytes > MAX_ARTIFACT_BYTES
    ):
        failures.append("quantized artifact must be at most 1GB")
    if (
        candidate.warm_p95_seconds is None
        or candidate.warm_p95_seconds > MAX_WARM_P95_SECONDS
    ):
        failures.append("Windows warm 256-token p95 must be at most 15 seconds")
    return failures


def wilson_lower_bound(wins: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (center - radius) / denominator


def deployment_failures(report: EvaluationReport) -> list[str]:
    total = int(report.comparisons.get("total", 0))
    wins = int(report.comparisons.get("candidate_wins", 0))
    failures: list[str] = []
    if total < 100:
        failures.append("deployment requires at least 100 blinded comparisons")
    if total <= 0 or wins / total <= 0.55:
        failures.append("candidate blinded win rate must exceed 55%")
    if wilson_lower_bound(wins, total) <= 0.50:
        failures.append("95% Wilson lower bound must exceed 50%")
    return failures


def select_causal_base(reports: Iterable[EvaluationReport]) -> EvaluationReport:
    """Pick by gates, semantic/structural quality, latency, then artifact size."""
    rows = list(reports)
    if not rows:
        raise ValueError("causal base bakeoff requires reports")
    return max(
        rows,
        key=lambda report: (
            report.ship_gates_pass,
            report.metrics.get("semantic_score", 0.0)
            + report.metrics.get("structural_similarity", 0.0),
            -float(
                math.inf if report.warm_p95_seconds is None else report.warm_p95_seconds
            ),
            -int(
                MAX_ARTIFACT_BYTES + 1
                if report.artifact_size_bytes is None
                else report.artifact_size_bytes
            ),
        ),
    )
