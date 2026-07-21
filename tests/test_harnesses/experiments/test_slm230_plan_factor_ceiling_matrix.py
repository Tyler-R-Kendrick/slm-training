"""Tests for the SLM-230 (SPV0-04) plan-factor oracle-substitution ceiling
matrix."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm230_plan_factor_ceiling_matrix import (
    EXPERIMENT_ID,
    MATRIX_SET,
    Slm230Report,
    build_default_arms,
    render_markdown,
    run_fixture_matrix,
)


def _row(report, arm_id):
    return next(r for r in report.rows if r.arm_id == arm_id)


def test_default_arms_shape() -> None:
    arms = build_default_arms()
    ids = {a.arm_id for a in arms}
    assert ids == {
        "C0_no_plan",
        "C1_roles_only",
        "C2_topology_only",
        "C3_bindings_only",
        "C4_roles_topology",
        "C5_roles_topology_bindings",
        "C6_full_gold_oracle",
    }
    only_promotable = {a.arm_id for a in arms if a.promotable}
    assert only_promotable == {"C0_no_plan"}


def test_fixture_runs_all_arms() -> None:
    report = run_fixture_matrix(corpus_size=8, corpus_seed=0)
    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.rows) == 7
    assert report.gate_hash
    # build_fixture_plan_corpus splits 80/20 train/val; only the train split
    # is used, so corpus_size is int(corpus_size_arg * 0.8).
    assert report.corpus_size == 6


def test_no_plan_baseline_never_produces_a_content_seed() -> None:
    report = run_fixture_matrix(corpus_size=8, corpus_seed=0)
    baseline = _row(report, "C0_no_plan")
    assert baseline.seed_valid_count == 0
    assert baseline.seed_valid_rate == 0.0


def test_isolated_single_factor_arms_do_not_beat_baseline() -> None:
    report = run_fixture_matrix(corpus_size=8, corpus_seed=0)
    baseline = _row(report, "C0_no_plan")
    for arm_id in ("C1_roles_only", "C2_topology_only", "C3_bindings_only"):
        row = _row(report, arm_id)
        assert row.seed_valid_rate <= baseline.seed_valid_rate, arm_id


def test_combined_roles_topology_reaches_full_validity() -> None:
    report = run_fixture_matrix(corpus_size=8, corpus_seed=0)
    combined = _row(report, "C4_roles_topology")
    assert combined.seed_valid_rate == 1.0
    assert combined.mean_component_coverage == 1.0


def test_bindings_raises_placeholder_attachment_over_structural_only() -> None:
    report = run_fixture_matrix(corpus_size=8, corpus_seed=0)
    structural = _row(report, "C4_roles_topology")
    with_bindings = _row(report, "C5_roles_topology_bindings")
    full_oracle = _row(report, "C6_full_gold_oracle")
    assert (structural.mean_placeholder_attachment_ratio or 0.0) == 0.0
    assert (with_bindings.mean_placeholder_attachment_ratio or 0.0) > 0.0
    # Archetype has no effect on PlanSeedBuilder output, so the full-oracle
    # arm should match the roles+topology+bindings arm exactly.
    assert full_oracle.seed_valid_rate == with_bindings.seed_valid_rate
    assert full_oracle.mean_placeholder_attachment_ratio == with_bindings.mean_placeholder_attachment_ratio


def test_cardinality_fields_are_confirmed_unpopulated() -> None:
    report = run_fixture_matrix(corpus_size=8, corpus_seed=0)
    assert report.cardinality_populated is False


def test_disposition_confirms_joint_requirement() -> None:
    report = run_fixture_matrix(corpus_size=8, corpus_seed=0)
    assert report.disposition == "ceiling_confirmed_joint_requirement"


def test_report_roundtrips_through_dict() -> None:
    report = run_fixture_matrix(corpus_size=8, corpus_seed=0)
    payload = report.to_dict()
    restored = Slm230Report.from_dict(payload)
    assert restored.to_dict() == payload


def test_gate_hash_is_deterministic() -> None:
    a = run_fixture_matrix(corpus_size=8, corpus_seed=0)
    b = run_fixture_matrix(corpus_size=8, corpus_seed=0)
    assert a.gate_hash == b.gate_hash
    assert a.disposition == b.disposition


def test_render_markdown_includes_disposition_and_table() -> None:
    report = run_fixture_matrix(corpus_size=8, corpus_seed=0)
    text = render_markdown(report)
    assert report.disposition in text
    assert "| arm | factors | n |" in text
    assert "No-go for promotion" in text
    assert "C0_no_plan" in text
