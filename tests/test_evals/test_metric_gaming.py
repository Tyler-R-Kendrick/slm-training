"""Regression tests for the SDE0-02 metric-gaming stress suite."""

from __future__ import annotations

import json

import pytest

from slm_training.evals.metric_gaming import (
    ALL_SLICES,
    SLICE_CANARY_AST_SIMILAR_MISSING_COMPONENT,
    SLICE_CANARY_CANONICAL_EQUIVALENT_POSITIVE,
    SLICE_CANARY_OVERLONG_ECONOMY_VIOLATION,
    SLICE_CANARY_RENDER_SEMANTICS_MISMATCH,
    SLICE_CANARY_RIGHT_INVENTORY_WRONG_HIERARCHY,
    SLICE_CANARY_RIGHT_ROLE_WRONG_BINDING,
    SLICE_INVENTORY_FREE,
    SLICE_MINIMAL_VALID,
    SLICE_RARE_OMISSION,
    SLICE_RETRY_SENSITIVE,
    MetricGamingCase,
    build_all_cases,
    build_inventory_free_binding_cases,
    build_retry_sensitive_cases,
    evaluate_all_retry_cases,
    evaluate_metric_gaming,
    evaluate_retry_attempts,
    write_manifest,
)


@pytest.fixture
def all_cases() -> list[MetricGamingCase]:
    return build_all_cases(seed=0)


def test_build_all_cases_count_and_slices(all_cases: list[MetricGamingCase]) -> None:
    assert len(all_cases) == 119
    by_slice = {}
    for case in all_cases:
        by_slice.setdefault(case.slice, 0)
        by_slice[case.slice] += 1
    assert set(by_slice) == set(ALL_SLICES)
    assert by_slice[SLICE_MINIMAL_VALID] == 35
    assert by_slice[SLICE_RARE_OMISSION] == 12
    assert by_slice[SLICE_INVENTORY_FREE] == 26
    assert by_slice[SLICE_RETRY_SENSITIVE] == 28
    assert by_slice[SLICE_CANARY_RIGHT_ROLE_WRONG_BINDING] == 3
    assert by_slice[SLICE_CANARY_RIGHT_INVENTORY_WRONG_HIERARCHY] == 3
    assert by_slice[SLICE_CANARY_RENDER_SEMANTICS_MISMATCH] == 3
    assert by_slice[SLICE_CANARY_AST_SIMILAR_MISSING_COMPONENT] == 3
    assert by_slice[SLICE_CANARY_OVERLONG_ECONOMY_VIOLATION] == 3
    assert by_slice[SLICE_CANARY_CANONICAL_EQUIVALENT_POSITIVE] == 3


def test_all_preds_are_parser_schema_valid(all_cases: list[MetricGamingCase]) -> None:
    from slm_training.dsl.parser import ParseError, validate

    failures: list[str] = []
    for case in all_cases:
        try:
            validate(case.pred_openui)
        except (ParseError, RuntimeError, ValueError) as exc:
            failures.append(f"{case.id}: {exc}")
    assert not failures, "\n".join(failures)


def test_expected_verdicts_match_binding_aware_meaningful_v2(
    all_cases: list[MetricGamingCase],
) -> None:
    report = evaluate_metric_gaming(all_cases)
    assert report.false_positive_count == 0
    assert report.false_negative_count == 0
    assert report.strict_rate == 1.0
    for sc in report.cases:
        assert sc.report.verdict is sc.case.expected_verdict, sc.case.id


def test_expected_reason_substrings_appear(all_cases: list[MetricGamingCase]) -> None:
    report = evaluate_metric_gaming(all_cases)
    failures: list[str] = []
    for sc in report.cases:
        if not sc.case.expected_reason_substrings:
            continue
        reasons = set(sc.report.reason_codes)
        missing = [
            sub for sub in sc.case.expected_reason_substrings if sub not in reasons
        ]
        if missing:
            failures.append(
                f"{sc.case.id}: missing reason(s) {missing}; got {reasons}"
            )
    assert not failures, "\n".join(failures)


def test_inventory_on_vs_off_diverge() -> None:
    on_by_id = {}
    off_by_id = {}
    for case in build_inventory_free_binding_cases(seed=0):
        if case.id.endswith("_inventory_on"):
            on_by_id[case.id.replace("_inventory_on", "")] = case
        if case.id.endswith("_inventory_off"):
            off_by_id[case.id.replace("_inventory_off", "")] = case
    assert on_by_id and off_by_id
    for base_id in set(on_by_id) & set(off_by_id):
        on = on_by_id[base_id]
        off = off_by_id[base_id]
        assert on.pred_openui == off.pred_openui
        assert on.expected_verdict is True
        assert off.expected_verdict is False


def test_retry_first_selected_oracle_metrics_differ(all_cases: list[MetricGamingCase]) -> None:
    retry_results = evaluate_all_retry_cases(all_cases)
    assert retry_results
    # Every positive retry case must have a passing oracle; the first attempt
    # can be worse than the selected/best attempt.
    for row in retry_results:
        if not row["case_id"].endswith("_retry_all_fail"):
            assert row["oracle_best_pass"] is True
            assert row["selected_attempt_pass"] is True
        else:
            assert row["oracle_best_pass"] is False

    differing = sum(
        1
        for row in retry_results
        if row["first_attempt_pass"] != row["selected_attempt_pass"]
        or row["first_attempt_pass"] != row["oracle_best_pass"]
    )
    # At least some rows expose a first-to-best gap (fixture is 50%).
    assert differing >= len(retry_results) * 0.4


def test_retry_attempt_selector_custom() -> None:
    cases = build_retry_sensitive_cases(seed=0)
    case = next(c for c in cases if c.id.endswith("_retry_2"))
    attempts = json.loads(case.gold_openui or "[]")
    assert isinstance(attempts, list) and len(attempts) == 2
    result = evaluate_retry_attempts(
        case, attempts, selector=lambda scored: len(scored) - 1
    )
    assert result["selected_attempt_index"] == len(attempts) - 1
    assert result["oracle_best_pass"] is True


def test_retry_all_fail_case_is_negative(all_cases: list[MetricGamingCase]) -> None:
    report = evaluate_metric_gaming(all_cases)
    for sc in report.cases:
        if sc.case.id.endswith("_retry_all_fail"):
            assert sc.case.expected_verdict is False
            assert sc.report.verdict is False


def test_manifest_round_trip(tmp_path, all_cases: list[MetricGamingCase]) -> None:
    report = evaluate_metric_gaming(all_cases)
    retry_results = evaluate_all_retry_cases(all_cases)
    path = tmp_path / "manifest.json"
    write_manifest(report, retry_results, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == report.schema_version
    assert payload["metric_gaming_report"]["n_cases"] == report.n_cases
    assert len(payload["retry_results"]) == len(retry_results)


def test_case_to_dict_round_trip(all_cases: list[MetricGamingCase]) -> None:
    for case in all_cases:
        d = case.to_dict()
        assert d["id"] == case.id
        assert d["slice"] == case.slice
        assert "request" in d
        assert d["request"]["prompt"] == case.request.prompt


def test_canary_slices_are_present_and_have_expected_reasons(
    all_cases: list[MetricGamingCase],
) -> None:
    report = evaluate_metric_gaming(all_cases)
    canary_slices = {
        SLICE_CANARY_RIGHT_ROLE_WRONG_BINDING,
        SLICE_CANARY_RIGHT_INVENTORY_WRONG_HIERARCHY,
        SLICE_CANARY_RENDER_SEMANTICS_MISMATCH,
        SLICE_CANARY_AST_SIMILAR_MISSING_COMPONENT,
        SLICE_CANARY_OVERLONG_ECONOMY_VIOLATION,
        SLICE_CANARY_CANONICAL_EQUIVALENT_POSITIVE,
    }
    for slice_name in canary_slices:
        assert slice_name in report.slices, f"missing slice {slice_name}"
        slice_report = report.slices[slice_name]
        assert slice_report.n > 0, f"slice {slice_name} has no cases"

    # Canonical-equivalent positives should all pass.
    for sc in report.cases:
        if sc.case.slice == SLICE_CANARY_CANONICAL_EQUIVALENT_POSITIVE:
            assert sc.case.expected_verdict is True
            assert sc.report.verdict is True, sc.case.id

    # Other canary cases should all fail.
    for sc in report.cases:
        if sc.case.slice in canary_slices - {SLICE_CANARY_CANONICAL_EQUIVALENT_POSITIVE}:
            assert sc.case.expected_verdict is False
            assert sc.report.verdict is False, sc.case.id
            assert sc.case.expected_reason_substrings
            reasons = set(sc.report.reason_codes)
            missing = [
                sub for sub in sc.case.expected_reason_substrings if sub not in reasons
            ]
            assert not missing, f"{sc.case.id}: missing reason(s) {missing}; got {reasons}"
