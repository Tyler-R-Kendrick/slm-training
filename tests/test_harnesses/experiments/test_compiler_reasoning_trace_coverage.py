"""Reproducibility tests for the LOT0-02 bounded K/c coverage probe.

Test 7 (SLM-249 acceptance): "K/c coverage calculations reproduce from the
corpus" -- here, the bounded fixture probe.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.lang_core import bridge_available
from slm_training.harnesses.experiments.compiler_reasoning_trace_coverage import (
    DEFAULT_PROBE_PATH,
    build_bounded_coverage_report,
    render_markdown,
)

SKIP_REASON = "OpenUI bridge deps missing"


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_coverage_report_reproduces_deterministically() -> None:
    first = build_bounded_coverage_report()
    second = build_bounded_coverage_report()
    assert first == second


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_coverage_report_is_labeled_bounded_probe_over_the_committed_fixture_set() -> None:
    report = build_bounded_coverage_report()
    assert report.evidence_class == "bounded_probe"
    assert report.sample_size == 16
    assert DEFAULT_PROBE_PATH.name == "test_seeds.jsonl"
    assert report.proposed_k == 6
    assert len(report.stage_stats) == 6


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_no_truncation_occurred_over_the_bounded_probe() -> None:
    report = build_bounded_coverage_report()
    assert report.fraction_truncated == 0.0
    assert "no truncation occurred" in report.truncation_policy


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_markdown_render_includes_evidence_class_and_sample_size() -> None:
    report = build_bounded_coverage_report()
    md = render_markdown(report)
    assert "bounded_probe" in md
    assert f"n={report.sample_size}" in md
