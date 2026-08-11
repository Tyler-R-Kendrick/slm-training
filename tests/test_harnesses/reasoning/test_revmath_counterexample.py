"""HARN-06 / SLM-546: computable/finite counterexample validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.judgment import JudgmentOutcome
from slm_training.harnesses.reasoning.revmath.counterexample import (
    check_counterexample_against_theorem,
    evaluate_counterexample,
    parse_counterexample_meta,
)
from slm_training.harnesses.reasoning.revmath.plugins import (
    CounterexamplePlugin,
    default_plugin_registry,
    resolve_plugin,
)
from slm_training.harnesses.reasoning.revmath.runner import run_revmath_task
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathSchemaError,
    RevmathTaskV1,
    parse_revmath_task,
)

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "src" / "slm_training" / "resources" / "revmath" / "fixtures"


def _load_pair(stem: str) -> tuple[RevmathTaskV1, object]:
    task = parse_revmath_task(
        json.loads((FIXTURES / f"{stem}.task.json").read_text(encoding="utf-8"))
    )
    meta = parse_counterexample_meta(
        json.loads((FIXTURES / f"{stem}.meta.json").read_text(encoding="utf-8"))
    )
    return task, meta


def _run(task: RevmathTaskV1, meta: object):
    def run_check(baseline: RevmathTaskV1, _meta):
        plugin = CounterexamplePlugin(meta_override=meta)  # type: ignore[arg-type]
        registry = default_plugin_registry()
        registry["computable_finite_counterexample"] = plugin
        return run_revmath_task(baseline, hermetic=True, plugin_registry=registry)

    return evaluate_counterexample(task, meta, run_check=run_check)  # type: ignore[arg-type]


def test_plugin_registered() -> None:
    task, _meta = _load_pair("counterexample_checked")
    plugin = resolve_plugin(task)
    assert plugin is not None
    assert plugin.task_kind == "computable_finite_counterexample"


def test_checked_counterexample_refutes_weakened_theorem() -> None:
    task, meta = _load_pair("counterexample_checked")
    check = check_counterexample_against_theorem(meta)  # type: ignore[arg-type]
    assert check.independently_checked is True
    assert check.matches_proposition is True
    assert check.matches_assumptions is True
    report = _run(task, meta)
    assert report.outcome == "refuted"
    assert report.judgment["outcome"] == JudgmentOutcome.REFUTED.value
    assert report.to_dict()["inferred_impossibility_from_search_failure"] is False
    assert report.check.independently_checked is True
    assert list(report.weakened_assumption_ids) == ["ISigma1"]


def test_search_failed_is_unknown_not_refuted() -> None:
    task, meta = _load_pair("counterexample_search_failed")
    check = check_counterexample_against_theorem(meta)  # type: ignore[arg-type]
    assert check.independently_checked is False
    report = _run(task, meta)
    assert report.outcome == "unknown"
    assert report.judgment["outcome"] == JudgmentOutcome.UNKNOWN.value
    assert report.to_dict()["inferred_impossibility_from_search_failure"] is False


def test_no_counterexample_is_unknown() -> None:
    task, meta = _load_pair("counterexample_no_counterexample")
    report = _run(task, meta)
    assert report.outcome == "unknown"
    assert report.check.model_present is False


def test_mismatched_proposition_not_accepted() -> None:
    task, meta = _load_pair("counterexample_mismatch_prop")
    check = check_counterexample_against_theorem(meta)  # type: ignore[arg-type]
    assert check.independently_checked is False
    assert "model_targets_different_proposition" in check.rejection_reasons
    report = _run(task, meta)
    assert report.outcome in ("invalid", "unknown")
    assert report.outcome != "refuted"


def test_changed_proposition_rejected_at_meta_match() -> None:
    _task, meta = _load_pair("counterexample_checked")
    payload = json.loads(
        (FIXTURES / "counterexample_checked.task.json").read_text(encoding="utf-8")
    )
    payload["proposition"]["statement"] = "forall n, n = n"
    from slm_training.harness_core.lineage.records import content_sha

    payload["proposition"]["statement_sha256"] = content_sha("forall n, n = n")
    broken = parse_revmath_task(payload)
    with pytest.raises(RevmathSchemaError, match="weakened theorem"):
        evaluate_counterexample(
            broken,
            meta,  # type: ignore[arg-type]
            run_check=lambda *_a, **_k: None,
        )
