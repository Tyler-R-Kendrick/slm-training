"""Tests for the SLM-237 (PCR0-01) placeholder-contract literal-content
robustness probe."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm237_placeholder_contract_literal_content_robustness import (
    EXPERIMENT_ID,
    GOLD_PLACEHOLDER,
    MATRIX_SET,
    ContentVariant,
    PlaceholderContractRobustnessReport,
    build_default_variants,
    render_markdown,
    run_robustness_fixture,
)


def _rows(report, *, shape=None, category=None):
    rows = report.rows
    if shape is not None:
        rows = [r for r in rows if r.shape == shape]
    if category is not None:
        rows = [r for r in rows if r.category == category]
    return rows


def test_default_variants_shape() -> None:
    variants = build_default_variants()
    names = {v.name for v in variants}
    assert "neutral" in names
    assert "violation_mentioned" in names
    assert "violation_unmentioned" in names
    categories = {v.category for v in variants}
    assert categories == {
        "baseline",
        "content_only",
        "spurious_unrelated",
        "contract_violation_mentioned",
        "contract_violation_unmentioned",
    }


def test_fixture_runs_all_shape_variant_pairs() -> None:
    report = run_robustness_fixture()
    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    n_shapes = 3
    n_variants = len(build_default_variants())
    assert len(report.rows) == n_shapes * n_variants
    assert report.gate_hash


def test_baseline_rows_score_perfect_and_unviolated() -> None:
    report = run_robustness_fixture()
    baselines = _rows(report, category="baseline")
    assert baselines
    for row in baselines:
        assert row.contract_violated is False
        assert row.contract_precision == 1.0
        assert row.contract_recall == 1.0
        assert row.placeholder_fidelity == 1.0
        assert row.pred_placeholder_tokens == (GOLD_PLACEHOLDER,)


def test_content_only_wording_edit_is_robust() -> None:
    report = run_robustness_fixture()
    rows = _rows(report, category="content_only")
    assert rows
    for row in rows:
        assert row.contract_precision == 1.0
        assert row.contract_recall == 1.0
        assert row.placeholder_fidelity == 1.0


def test_spurious_unrelated_mention_lowers_precision_only() -> None:
    report = run_robustness_fixture()
    rows = _rows(report, category="spurious_unrelated")
    assert rows
    for row in rows:
        assert row.contract_precision is not None
        assert row.contract_precision < 1.0
        # Recall / fidelity only test the gold set, so they stay perfect.
        assert row.contract_recall == 1.0
        assert row.placeholder_fidelity == 1.0
        assert GOLD_PLACEHOLDER in row.pred_placeholder_tokens
        assert ":support.email" in row.pred_placeholder_tokens


def test_contract_violation_mentioned_gets_false_credit() -> None:
    report = run_robustness_fixture()
    rows = _rows(report, category="contract_violation_mentioned")
    assert rows
    for row in rows:
        assert row.contract_violated is True
        assert row.false_credit is True
        assert row.contract_precision == 1.0
        assert row.contract_recall == 1.0
        assert row.placeholder_fidelity == 1.0


def test_contract_violation_unmentioned_is_correctly_penalized() -> None:
    report = run_robustness_fixture()
    rows = _rows(report, category="contract_violation_unmentioned")
    assert rows
    for row in rows:
        assert row.contract_violated is True
        assert row.false_credit is False
        assert row.contract_precision == 0.0
        assert row.contract_recall == 0.0
        assert row.placeholder_fidelity == 0.0
        assert row.pred_placeholder_tokens == ()


def test_mentioned_vs_unmentioned_violation_scores_diverge() -> None:
    report = run_robustness_fixture()
    mentioned = {r.shape: r for r in _rows(report, category="contract_violation_mentioned")}
    unmentioned = {r.shape: r for r in _rows(report, category="contract_violation_unmentioned")}
    assert mentioned.keys() == unmentioned.keys()
    for shape in mentioned:
        # Same real violation (title hardcoded, no real placeholder used);
        # score flips purely on an incidental literal-text mention.
        assert mentioned[shape].contract_violated == unmentioned[shape].contract_violated
        assert mentioned[shape].placeholder_fidelity == 1.0
        assert unmentioned[shape].placeholder_fidelity == 0.0


def test_disposition_confirms_the_gap() -> None:
    report = run_robustness_fixture()
    assert report.disposition == "gap_confirmed"
    assert "false credit" in report.disposition_rationale.lower()


def test_reward_probe_reflects_placeholder_fidelity_term() -> None:
    report = run_robustness_fixture()
    by_variant = {r.variant: r for r in report.reward_probe_rows}
    mentioned = by_variant["violation_mentioned"]
    unmentioned = by_variant["violation_unmentioned"]
    baseline = by_variant["neutral"]
    # Both violation variants are real contract violations, but the reward
    # contract's placeholder_fidelity term flips purely on incidental text.
    assert mentioned.contract_violated is True
    assert unmentioned.contract_violated is True
    assert mentioned.placeholder_fidelity == 1.0
    assert unmentioned.placeholder_fidelity == 0.0
    assert mentioned.placeholder_fidelity == baseline.placeholder_fidelity
    assert mentioned.composite > unmentioned.composite


def test_report_roundtrips_through_dict() -> None:
    report = run_robustness_fixture()
    payload = report.to_dict()
    restored = PlaceholderContractRobustnessReport.from_dict(payload)
    assert restored.to_dict() == payload


def test_gate_hash_is_deterministic() -> None:
    a = run_robustness_fixture()
    b = run_robustness_fixture()
    assert a.gate_hash == b.gate_hash
    assert a.disposition == b.disposition


def test_custom_variant_list_is_respected() -> None:
    variants = [
        ContentVariant("neutral", "baseline", True, "", "Contact our team today"),
        ContentVariant(
            "only_variant", "content_only", True, "", "Reach out to support now"
        ),
    ]
    report = run_robustness_fixture(variants=variants)
    assert len(report.rows) == 3 * len(variants)
    assert {r.variant for r in report.rows} == {"neutral", "only_variant"}


def test_render_markdown_includes_disposition_and_tables() -> None:
    report = run_robustness_fixture()
    text = render_markdown(report)
    assert report.disposition in text
    assert "| shape | variant | category | violated |" in text
    assert "Downstream RL reward probe" in text
    assert "No-go for promotion" in text
    assert "violation_mentioned" in text
