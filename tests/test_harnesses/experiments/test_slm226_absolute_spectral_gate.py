"""Tests for the SLM-226 absolute spectral target gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.harnesses.experiments.slm226_absolute_spectral_gate import (
    build_width_manifest,
    decide_absolute_gate,
    render_markdown,
    require_absolute_spectral_gate,
    run_absolute_spectral_boundary,
)


def test_width_manifest_is_frozen_by_role_and_shape() -> None:
    specs = build_width_manifest()
    assert [row.width for row in specs] == [128, 256, 512]
    assert [row.shape_id for row in specs] == ["128x128", "256x128", "512x128"]
    assert {row.role for row in specs} == {"ctx_proj"}


def test_gate_fails_closed_without_causal_or_checkpoint_evidence() -> None:
    gate = decide_absolute_gate(
        causal_shape_effect_supported=False,
        durable_checkpoint_families=0,
        semantic_floor_verdict="inconclusive",
    )
    assert gate.verdict == "descriptive_only"
    assert gate.authorized_shapes == ()
    assert set(gate.blocked_interventions) == {"ww_pgd", "trace_log", "alpha_target"}
    with pytest.raises(RuntimeError, match="not authorized"):
        require_absolute_spectral_gate(
            gate,
            claim_or_intervention="ww_pgd",
            role="ctx_proj",
            shape="128x128",
        )


def test_unknown_intervention_fails_validation() -> None:
    gate = decide_absolute_gate(
        causal_shape_effect_supported=False,
        durable_checkpoint_families=0,
        semantic_floor_verdict="inconclusive",
    )
    with pytest.raises(ValueError, match="unknown"):
        require_absolute_spectral_gate(
            gate,
            claim_or_intervention="raw_alpha",
            role="ctx_proj",
            shape="128x128",
        )


def test_bounded_report_contains_null_trained_and_control_rows() -> None:
    root = Path(__file__).resolve().parents[3]
    report = run_absolute_spectral_boundary(
        repo_root=root,
        seeds=(0,),
        null_draws=5,
    )
    assert len(report.null_shapes) == 3
    assert len(report.trained_shapes) == 3
    assert len(report.controls) == 6
    assert report.gate.verdict == "descriptive_only"
    assert all(row.draws == 5 for row in report.null_shapes)
    markdown = render_markdown(report)
    assert "proximity to alpha=2 is not an authorization signal" in markdown
    assert "No canonical model evaluation or AgentV run" in markdown
