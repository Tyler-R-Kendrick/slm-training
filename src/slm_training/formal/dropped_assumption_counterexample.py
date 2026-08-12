"""RESEARCH-13 / SLM-568: formal counterexample under dropped assumptions (default-off).

Fixture-only comparison of proof-search terminal labels vs bounded counterexample
synthesis with HARN-06 independent validation on frozen revmath tasks.
Never promotes search failure to refutation (KERN-01).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.counterexample import (
    evaluate_counterexample,
    parse_counterexample_meta,
)
from slm_training.harnesses.reasoning.revmath.plugins import (
    CounterexamplePlugin,
    default_plugin_registry,
)
from slm_training.harnesses.reasoning.revmath.runner import run_revmath_task
from slm_training.harnesses.reasoning.revmath.schemas import parse_revmath_task

BACKEND_ID = "dropped_assumption_counterexample_pilot"
TOOLCHAIN_ID = "hermetic_python_dropped_assumption_cex_v1"
CORPUS_RELATIVE = "src/slm_training/resources/formal/dropped_assumption_tasks.v1.json"
FIXTURES_RELATIVE = "src/slm_training/resources/revmath/fixtures"

NecessityLabel = Literal["necessary", "redundant", "unknown", "invalid"]


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "fixtures_relpath": FIXTURES_RELATIVE,
        "trust_domain": "research_only/dropped_assumption_counterexample",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class DroppedAssumptionTaskV1:
    task_id: str
    fixture_stem: str
    ground_truth_necessity: NecessityLabel
    expects_independent_check: bool
    notes: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DroppedAssumptionTaskV1:
        return cls(
            task_id=str(data["task_id"]),
            fixture_stem=str(data["fixture_stem"]),
            ground_truth_necessity=str(data["ground_truth_necessity"]),  # type: ignore[arg-type]
            expects_independent_check=bool(data["expects_independent_check"]),
            notes=str(data.get("notes", "")),
        )


def load_corpus(*, root: Path | None = None) -> dict[str, Any]:
    path = (root or _repo_root()) / CORPUS_RELATIVE
    return json.loads(path.read_text(encoding="utf-8"))


def default_tasks(*, root: Path | None = None) -> tuple[DroppedAssumptionTaskV1, ...]:
    raw = load_corpus(root=root)
    return tuple(DroppedAssumptionTaskV1.from_dict(t) for t in raw["tasks"])


def _load_fixture_pair(stem: str, *, root: Path) -> tuple[Any, Any]:
    fixtures = root / FIXTURES_RELATIVE
    task = parse_revmath_task(
        json.loads((fixtures / f"{stem}.task.json").read_text(encoding="utf-8"))
    )
    meta = parse_counterexample_meta(
        json.loads((fixtures / f"{stem}.meta.json").read_text(encoding="utf-8"))
    )
    return task, meta


def _run_counterexample(task: Any, meta: Any) -> Any:
    def run_check(baseline: Any, bound_meta: Any):
        plugin = CounterexamplePlugin(meta_override=bound_meta)
        registry = default_plugin_registry()
        registry["computable_finite_counterexample"] = plugin
        return run_revmath_task(baseline, hermetic=True, plugin_registry=registry)

    return evaluate_counterexample(task, meta, run_check=run_check)


def _control_necessity_label(meta: Any, report: Any | None) -> NecessityLabel:
    """Proof-search terminal only — may falsely claim necessary on search ok."""

    if meta.search_status in ("failed", "timeout", "incomplete", "no_result"):
        return "unknown"
    if report is not None and report.outcome == "refuted":
        return "necessary"
    if report is not None and report.outcome == "invalid":
        return "invalid"
    return "unknown"


def _treatment_necessity_label(report: Any) -> NecessityLabel:
    if report.check.independently_checked and report.outcome == "refuted":
        return "necessary"
    if report.outcome == "invalid":
        return "invalid"
    return "unknown"


def evaluate_control(record: DroppedAssumptionTaskV1, *, root: Path) -> dict[str, Any]:
    task, meta = _load_fixture_pair(record.fixture_stem, root=root)
    report = _run_counterexample(task, meta)
    label = _control_necessity_label(meta, report)
    return {
        "arm": "proof_search_terminal",
        "task_id": record.task_id,
        "fixture_stem": record.fixture_stem,
        "necessity_label": label,
        "search_status": meta.search_status,
        "outcome": report.outcome,
        "independently_checked": report.check.independently_checked,
    }


def evaluate_treatment(record: DroppedAssumptionTaskV1, *, root: Path) -> dict[str, Any]:
    task, meta = _load_fixture_pair(record.fixture_stem, root=root)
    report = _run_counterexample(task, meta)
    label = _treatment_necessity_label(report)
    return {
        "arm": "bounded_counterexample_validation",
        "task_id": record.task_id,
        "fixture_stem": record.fixture_stem,
        "necessity_label": label,
        "search_status": meta.search_status,
        "outcome": report.outcome,
        "independently_checked": report.check.independently_checked,
    }


def paired_counterexample_report(*, root: Path | None = None) -> dict[str, Any]:
    base = root or _repo_root()
    raw = load_corpus(root=base)
    records = default_tasks(root=base)

    control_rows = [evaluate_control(r, root=base) for r in records]
    treatment_rows = [evaluate_treatment(r, root=base) for r in records]

    expect_check = [
        (rec, treat)
        for rec, treat in zip(records, treatment_rows, strict=True)
        if rec.expects_independent_check
    ]
    checked = sum(1 for _rec, treat in expect_check if treat["independently_checked"])
    primary = checked / max(len(expect_check), 1)

    search_failure_as_refutation = sum(
        1
        for rec, ctrl in zip(records, control_rows, strict=True)
        if rec.ground_truth_necessity == "unknown"
        and ctrl["necessity_label"] == "necessary"
    )

    model_check_disagreements = sum(
        1
        for rec, treat in zip(records, treatment_rows, strict=True)
        if rec.ground_truth_necessity == "necessary"
        and rec.expects_independent_check
        and not treat["independently_checked"]
    )

    assumption_set_drift = 0

    if search_failure_as_refutation > 0:
        decision = "reject"
        reason = "search_failure_labeled_as_necessary"
    elif model_check_disagreements > 0:
        decision = "reject"
        reason = "expected_independent_check_missing"
    elif primary < 1.0:
        decision = "reject"
        reason = "incomplete_independently_checked_counterexamples"
    else:
        decision = "accept"
        reason = "checked_finite_models_without_search_failure_refutation"

    return {
        "schema": "dropped_assumption_counterexample_report/v1",
        "independently_checked_finite_model_counterexample_rate": primary,
        "independent_check_success_count": checked,
        "expected_independent_check_count": len(expect_check),
        "search_failure_as_refutation_count": search_failure_as_refutation,
        "model_check_disagreements": model_check_disagreements,
        "assumption_set_drift": assumption_set_drift,
        "decision": decision,
        "reason": reason,
        "control_rows": control_rows,
        "treatment_rows": treatment_rows,
        "corpus_digest": content_sha(raw),
        "toolchain": pinned_toolchain(),
    }


__all__ = [
    "BACKEND_ID",
    "CORPUS_RELATIVE",
    "DroppedAssumptionTaskV1",
    "default_tasks",
    "evaluate_control",
    "evaluate_treatment",
    "load_corpus",
    "paired_counterexample_report",
    "pinned_toolchain",
]
