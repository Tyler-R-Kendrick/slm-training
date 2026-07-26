"""Tests for slm_training.harnesses.experiments.dca_causal_disposition (SLM-276)."""

from __future__ import annotations

from slm_training.harnesses.experiments.dca_causal_disposition import (
    NO_SHIP_READY,
    build_bundle,
    render_markdown,
)


def test_real_artifacts_yield_no_ship_ready() -> None:
    """Against the currently-committed artifacts every prerequisite leg is
    blocked/inconclusive, so the bundle records
    no_model_ship_ready_measurement_or_data_blocker, no eligible
    finalists, and no promotion.
    """
    bundle = build_bundle()

    assert bundle.outcome == NO_SHIP_READY
    assert bundle.promotion_action == "none_incumbent_retained"
    assert all(
        d["verdict"] in {"blocked", "unavailable_or_unknown"}
        for d in bundle.prerequisite_dispositions.values()
    )
    assert all(v.startswith("ineligible") for v in bundle.finalist_eligibility.values())
    assert bundle.evidence_gaps == []


def test_hypothesis_ledger_blocked_not_negative_claimed() -> None:
    """Blocked campaigns are recorded as blocked, never as tested-negative."""
    bundle = build_bundle()

    assert len(bundle.hypothesis_ledger) == 7
    assert all(h["disposition"] == "blocked" for h in bundle.hypothesis_ledger)
    assert all("no causal" in h["causal_interpretation"] for h in bundle.hypothesis_ledger)
    assert {h["hypothesis"] for h in bundle.hypothesis_ledger} == {
        f"DCA-H{i}" for i in range(1, 8)
    }


def test_missing_artifacts_fail_closed(tmp_path) -> None:
    bundle = build_bundle(tmp_path)

    assert bundle.outcome == NO_SHIP_READY
    assert bundle.evidence_gaps
    assert all(
        d["evidence_strength"] == "unavailable_or_unknown"
        for d in bundle.prerequisite_dispositions.values()
    )


def test_bundle_fingerprint_and_stamp() -> None:
    bundle = build_bundle()
    data = bundle.to_dict()

    assert data["bundle_fingerprint"]
    assert data["version_stamp"]["stamp_schema"] == "version_stamp/v1"


def test_render_markdown_covers_sections() -> None:
    markdown = render_markdown(build_bundle())

    assert "SLM-276" in markdown
    assert NO_SHIP_READY in markdown
    assert "Hypothesis ledger" in markdown
    assert "Finalist eligibility" in markdown
