"""Tests for the SLM-236 (SSR0-01) structural-similarity literal-content
robustness probe."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm236_structural_similarity_literal_content_robustness import (
    EXPERIMENT_ID,
    MATRIX_SET,
    ContentVariant,
    StructuralSimilarityRobustnessReport,
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
    assert "plain_alt_wording" in names
    assert "fake_component_single" in names
    categories = {v.category for v in variants}
    assert categories == {
        "baseline",
        "content_only",
        "benign_punctuation",
        "adversarial_regex",
        "adversarial_mixed",
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


def test_baseline_rows_match_exactly() -> None:
    report = run_robustness_fixture()
    baselines = _rows(report, category="baseline")
    assert baselines
    for row in baselines:
        assert row.structural_similarity == 1.0
        assert row.divergent is False
        assert row.spurious_component_keys == ()


def test_content_only_wording_edit_is_robust() -> None:
    report = run_robustness_fixture()
    rows = _rows(report, category="content_only")
    assert rows
    for row in rows:
        assert row.structural_similarity == 1.0
        assert row.divergent is False


def test_adversarial_regex_edit_lowers_score_and_flags_spurious_component() -> None:
    report = run_robustness_fixture()
    rows = [
        r
        for r in report.rows
        if r.category == "adversarial_regex" and r.variant == "fake_component_single"
    ]
    assert rows
    for row in rows:
        assert row.divergent is True
        assert row.structural_similarity < 1.0
        assert "Details" in row.spurious_component_keys


def test_benign_punctuation_edit_also_diverges() -> None:
    report = run_robustness_fixture()
    rows = _rows(report, category="benign_punctuation")
    assert rows
    for row in rows:
        # No fake component keys expected -- divergence here is the depth
        # (bracket/paren character count) proxy alone, not the regex.
        assert row.spurious_component_keys == ()
        assert row.divergent is True
        assert row.structural_similarity < 1.0


def test_disposition_confirms_the_gap() -> None:
    report = run_robustness_fixture()
    assert report.disposition == "gap_confirmed"
    assert "content-only" in report.disposition_rationale.lower()


def test_reward_probe_reflects_structural_term_only() -> None:
    report = run_robustness_fixture()
    by_variant = {r.variant: r for r in report.reward_probe_rows}
    neutral = by_variant["neutral"]
    adversarial = by_variant["fake_component_multi"]
    # parse validity and placeholder fidelity are held constant...
    assert adversarial.parse == neutral.parse
    assert adversarial.placeholder_fidelity == neutral.placeholder_fidelity
    # ...but composite reward still drops, driven entirely by the
    # structural_similarity term.
    assert adversarial.structural_similarity < neutral.structural_similarity
    assert adversarial.composite < neutral.composite


def test_report_roundtrips_through_dict() -> None:
    report = run_robustness_fixture()
    payload = report.to_dict()
    restored = StructuralSimilarityRobustnessReport.from_dict(payload)
    assert restored.to_dict() == payload


def test_gate_hash_is_deterministic() -> None:
    a = run_robustness_fixture()
    b = run_robustness_fixture()
    assert a.gate_hash == b.gate_hash
    assert a.disposition == b.disposition


def test_custom_variant_list_is_respected() -> None:
    variants = [
        ContentVariant("neutral", "Contact our team today", "baseline"),
        ContentVariant("only_variant", "Reach out to support now", "content_only"),
    ]
    report = run_robustness_fixture(variants=variants)
    assert len(report.rows) == 3 * len(variants)
    assert {r.variant for r in report.rows} == {"neutral", "only_variant"}


def test_render_markdown_includes_disposition_and_tables() -> None:
    report = run_robustness_fixture()
    text = render_markdown(report)
    assert report.disposition in text
    assert "| shape | variant | category | score |" in text
    assert "Downstream RL reward probe" in text
    assert "No-go for promotion" in text
    assert "fake_component_single" in text
