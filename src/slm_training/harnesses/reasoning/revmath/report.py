"""Typed revmath reports (HARN-03) + labeling disclaimers (HARN-08)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from slm_training.harnesses.reasoning.revmath.labeling import (
    AnalysisKind,
    RmLabelingState,
    default_analysis_kind_for_task,
    report_labeling_disclaimer,
)
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathReportV1,
    RevmathResultV1,
    RevmathTaskV1,
)
from slm_training.versioning import build_version_stamp


def labeling_notes_for_tasks(
    tasks: Sequence[RevmathTaskV1],
    *,
    labeling_state: RmLabelingState = "unclassified",
) -> tuple[dict[str, Any], ...]:
    """Per-task analysis_kind + disclaimer (ablation ≠ genuine RM)."""

    notes: list[dict[str, Any]] = []
    for task in tasks:
        kind: AnalysisKind = default_analysis_kind_for_task(task.task_kind)
        notes.append(
            {
                "task_id": task.task_id,
                "task_kind": task.task_kind,
                "analysis_kind": kind,
                "labeling_state": labeling_state,
                "disclaimer": report_labeling_disclaimer(
                    analysis_kind=kind, labeling_state=labeling_state
                ),
            }
        )
    return tuple(notes)


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
    enable_four_axis_ledger: bool = False,
    four_axis_analysis: Any | None = None,
    four_axis_ledger: Any | None = None,
    computability_class_id: str | None = None,
    refinement_trace_hash: str | None = None,
    refinement_certificate_sha256: str | None = None,
    formal_authority_class: str | None = None,
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
    projection = None
    if enable_four_axis_ledger:
        from slm_training.harnesses.four_axis_ledger import project_four_axis_ledger

        # Prefer explicit analysis; else fold result four_axis rows when homogeneous.
        analysis = four_axis_analysis
        if analysis is None and four_axis_ledger is None and results:
            analysis = results[0].four_axis
        projection = project_four_axis_ledger(
            harness_family="reasoning/revmath",
            analysis=analysis,
            ledger=four_axis_ledger,
            computability_class_id=computability_class_id,
            refinement_trace_hash=refinement_trace_hash,
            refinement_certificate_sha256=refinement_certificate_sha256,
            formal_authority_class=formal_authority_class,  # type: ignore[arg-type]
        ).to_dict()
    return RevmathReportV1(
        report_id=report_id,
        corpus_id=next(iter(corpus_ids)),
        campaign_manifest_sha256=next(iter(manifests)),
        task_ids=task_ids,
        result_ids=tuple(result.result_id for result in results),
        repair_ids=tuple(repair_ids),
        judgment_counts=_judgment_counts(results),
        version_stamp=stamp,
        four_axis_ledger_projection=projection,
    )


__all__ = ["build_revmath_report", "labeling_notes_for_tasks"]
