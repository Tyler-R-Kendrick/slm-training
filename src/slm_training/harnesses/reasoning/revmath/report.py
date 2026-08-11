"""Typed revmath reports (HARN-03)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathReportV1,
    RevmathResultV1,
    RevmathTaskV1,
)
from slm_training.versioning import build_version_stamp


def _judgment_counts(results: Sequence[RevmathResultV1]) -> dict[str, int]:
    counts = {"witnessed": 0, "refuted": 0, "unknown": 0, "invalid": 0}
    for result in results:
        outcome = result.solver_judgment().outcome.value
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def build_revmath_report(
    *,
    report_id: str,
    tasks: Sequence[RevmathTaskV1],
    results: Sequence[RevmathResultV1],
    repair_ids: Sequence[str] = (),
    version_stamp: dict[str, Any] | None = None,
) -> RevmathReportV1:
    """Aggregate a campaign-bound report from frozen tasks + results.

    Does not mutate tasks, campaigns, or proposition identity. Repair ids are
    references only — repair execution is HARN-09.
    """

    if not tasks:
        raise ValueError("revmath report requires at least one task")
    corpus_ids = {task.corpus.corpus_id for task in tasks}
    if len(corpus_ids) != 1:
        raise ValueError(
            f"revmath report tasks span multiple corpora: {sorted(corpus_ids)}"
        )
    manifests = {task.campaign.campaign_manifest_sha256 for task in tasks}
    if len(manifests) != 1:
        raise ValueError(
            "revmath report tasks span multiple campaign manifests: "
            + ", ".join(sorted(manifests))
        )
    task_ids = tuple(task.task_id for task in tasks)
    result_by_task = {result.task_id: result for result in results}
    for task in tasks:
        result = result_by_task.get(task.task_id)
        if result is None:
            continue
        if result.task_identity_digest != task.identity_digest():
            raise ValueError(
                f"result {result.result_id!r} identity digest does not match "
                f"task {task.task_id!r} (runner must not alter proposition/campaign)"
            )
    stamp = version_stamp
    if stamp is None:
        stamp = build_version_stamp("harness.reasoning.revmath")
    return RevmathReportV1(
        report_id=report_id,
        corpus_id=next(iter(corpus_ids)),
        campaign_manifest_sha256=next(iter(manifests)),
        task_ids=task_ids,
        result_ids=tuple(result.result_id for result in results),
        repair_ids=tuple(repair_ids),
        judgment_counts=_judgment_counts(results),
        version_stamp=stamp,
    )


__all__ = ["build_revmath_report"]
