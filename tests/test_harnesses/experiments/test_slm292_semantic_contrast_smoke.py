"""SLM-292 (AP-010): fixture-scale semantic-contrast smoke harness tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.experiments.slm292_semantic_contrast_smoke import (
    MATRIX_SET,
    render_markdown,
    run_smoke,
)


def test_run_smoke_is_fixture_wiring_claim_class() -> None:
    report = run_smoke(steps=2, d_model=16, n_heads=2, batch_pairs=3)
    assert report.claim_class == "fixture_wiring"
    assert "NOT" in report.disclosure or "not" in report.disclosure
    assert report.control_arm.semantic_contrast_loss_weight == 0.0
    assert report.treatment_arm.semantic_contrast_loss_weight > 0.0


def test_run_smoke_matches_control_and_treatment_arms() -> None:
    report = run_smoke(steps=2, d_model=16, n_heads=2, batch_pairs=3)
    # Matched: identical everything except the objective weight.
    for field in ("seed", "steps", "d_model", "n_heads", "context_layers", "denoiser_layers"):
        assert getattr(report.control_arm, field) == getattr(
            report.treatment_arm, field
        )


def test_run_smoke_logs_required_fields_only_for_treatment() -> None:
    report = run_smoke(steps=2, d_model=16, n_heads=2, batch_pairs=3)
    control_rows = [s for s in report.steps if s.arm_id == "control"]
    treatment_rows = [s for s in report.steps if s.arm_id == "treatment"]
    assert len(control_rows) == 2
    assert len(treatment_rows) == 2
    assert all(row.semantic_contrast_loss is None for row in control_rows)
    for row in treatment_rows:
        assert row.semantic_contrast_loss is not None
        assert row.semantic_contrast_loss_weight is not None
        assert row.semantic_contrast_objective == "margin"
        assert row.semantic_contrast_margin is not None
        assert row.semantic_contrast_pairs == 3
        assert row.semantic_contrast_family_counts
        assert row.semantic_contrast_positive_distance_mean is not None
        assert row.semantic_contrast_negative_distance_mean is not None


def test_run_smoke_reports_per_mutation_effects() -> None:
    report = run_smoke(steps=3, d_model=16, n_heads=2, batch_pairs=4)
    assert report.mutation_effects
    families = {effect.family for effect in report.mutation_effects}
    assert families <= {"content", "binding", "contract"}
    for effect in report.mutation_effects:
        assert effect.n_samples > 0


def test_render_markdown_includes_disclosure_and_tables() -> None:
    report = run_smoke(steps=2, d_model=16, n_heads=2, batch_pairs=3)
    md = render_markdown(report.to_dict())
    assert MATRIX_SET.startswith("slm292")
    assert "fixture_wiring" in md
    assert "Matched control/treatment arms" in md
    assert "Per-mutation-family effect" in md
    assert "Required logged fields" in md
    assert "Follow-up" in md
