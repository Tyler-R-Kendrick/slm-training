"""HARN-07 / SLM-547: quantitative-bound extraction + validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.bound_ast import (
    BOUND_CLOSURE_LIVE_UPPER,
    BOUND_FINITE_SEARCH_PREFIX_TREE,
    assert_registered_bound_ast_id,
)
from slm_training.formal.judgment import JudgmentOutcome
from slm_training.harnesses.reasoning.revmath.plugins import (
    QuantitativeBoundPlugin,
    default_plugin_registry,
    load_quantitative_bound_meta_for_task,
    resolve_plugin,
)
from slm_training.harnesses.reasoning.revmath.quantitative_bound import (
    extract_quantitative_bound,
    parse_quantitative_bound_meta,
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
    meta = parse_quantitative_bound_meta(
        json.loads((FIXTURES / f"{stem}.meta.json").read_text(encoding="utf-8"))
    )
    return task, meta


def test_plugin_registered_for_quantitative_kind() -> None:
    registry = default_plugin_registry()
    assert isinstance(registry["quantitative_bound_extraction"], QuantitativeBoundPlugin)
    task, _ = _load_pair("quant_finite_search")
    assert resolve_plugin(task) is not None


def test_finite_search_extracts_validated_bound() -> None:
    task, meta = _load_pair("quant_finite_search")
    assert_registered_bound_ast_id(BOUND_FINITE_SEARCH_PREFIX_TREE)
    report = extract_quantitative_bound(task, meta)  # type: ignore[arg-type]
    assert report.outcome == "extracted"
    assert report.bound_ast_id == BOUND_FINITE_SEARCH_PREFIX_TREE
    assert report.bound_value == "15"
    assert report.theorem_derived_bound is True
    assert report.empirical_timing_claimed is False
    assert report.validated_via_bound_ast is True
    assert report.exhaustive_small_instance_ok is True
    assert report.certificate_sha256 is not None
    assert report.machine_model.wall_clock is False


def test_closure_live_upper_extracts_without_fabricating_kern05() -> None:
    task, meta = _load_pair("quant_closure_live_upper")
    assert_registered_bound_ast_id(BOUND_CLOSURE_LIVE_UPPER)
    report = extract_quantitative_bound(task, meta)  # type: ignore[arg-type]
    assert report.outcome == "extracted"
    assert report.bound_ast_id == BOUND_CLOSURE_LIVE_UPPER
    assert report.bound_value == "7"
    assert report.theorem_derived_bound is True
    assert report.empirical_timing_claimed is False
    assert "KERN-05" in report.detail or "closePass_subset" in report.detail


def test_nonextractable_lists_missing_assumptions_no_bound() -> None:
    task, meta = _load_pair("quant_nonextractable")
    report = extract_quantitative_bound(task, meta)  # type: ignore[arg-type]
    assert report.outcome == "nonextractable"
    assert report.bound_value is None
    assert report.bound_ast_id is None
    assert report.theorem_derived_bound is False
    assert report.empirical_timing_claimed is False
    assert "finite_parameter_dimension" in report.missing_assumptions
    assert "named_cost_model" in report.missing_assumptions
    assert report.validated_via_bound_ast is False


def test_wall_clock_machine_model_rejected() -> None:
    from slm_training.harnesses.reasoning.revmath.quantitative_bound import MachineModelV1

    with pytest.raises(RevmathSchemaError, match="wall_clock"):
        MachineModelV1(name="bad", wall_clock=True)


def test_hermetic_runner_finite_search_witnessed() -> None:
    task, _ = _load_pair("quant_finite_search")
    record = run_revmath_task(task, hermetic=True)
    assert record.result.solver_judgment().outcome is JudgmentOutcome.WITNESSED


def test_hermetic_runner_nonextractable_witnessed_refusal() -> None:
    """Refusal with listed missing assumptions is a successful check, not a fabricated bound."""

    task, _ = _load_pair("quant_nonextractable")
    record = run_revmath_task(task, hermetic=True)
    assert record.result.solver_judgment().outcome is JudgmentOutcome.WITNESSED
    meta = load_quantitative_bound_meta_for_task(task)
    assert meta is not None
    report = extract_quantitative_bound(task, meta)
    assert report.outcome == "nonextractable"
    assert report.bound_value is None


def test_meta_load_by_task_prefix() -> None:
    task, meta = _load_pair("quant_closure_live_upper")
    loaded = load_quantitative_bound_meta_for_task(task)
    assert loaded is not None
    assert loaded.meta_id == meta.meta_id  # type: ignore[union-attr]
