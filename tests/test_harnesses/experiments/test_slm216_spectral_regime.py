"""Tests for the SLM-216 spectral-regime matrix."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from slm_training.harnesses.experiments.slm216_spectral_regime import (
    DEFAULT_TARGET_TOKENS,
    build_fixture_matrix,
    decide_regime_gate,
    render_markdown,
    run_regime_cell,
    validate_matrix,
)


def test_matrix_is_preregistered_and_token_matched() -> None:
    specs = build_fixture_matrix()
    assert validate_matrix(specs) == []
    assert {row.target_tokens for row in specs} == {DEFAULT_TARGET_TOKENS}
    assert len(specs) == 6


def test_matrix_rejects_token_budget_drift() -> None:
    specs = list(build_fixture_matrix())
    specs[-1] = replace(specs[-1], target_tokens=640, snapshot_tokens=(0, 640))
    assert "fixed token budget" in " ".join(validate_matrix(specs))


def test_physical_and_effective_batch_are_recorded() -> None:
    spec = next(row for row in build_fixture_matrix() if row.accumulation > 1)
    result = run_regime_cell(spec, seed=0, null_draws=3)
    assert result.physical_batch == 2
    assert result.accumulation == 4
    assert result.effective_batch == 8
    assert result.target_tokens == DEFAULT_TARGET_TOKENS


def test_deferred_snapshots_do_not_change_training_result() -> None:
    spec = build_fixture_matrix()[0]
    with_snapshots = run_regime_cell(spec, seed=1, null_draws=3)
    final_only = run_regime_cell(
        spec,
        seed=1,
        null_draws=3,
        capture_intermediate=False,
    )
    assert with_snapshots.final_state_hash == final_only.final_state_hash


def test_duplicate_control_has_fewer_unique_records() -> None:
    diverse = run_regime_cell(build_fixture_matrix()[3], seed=0, null_draws=3)
    duplicated = run_regime_cell(build_fixture_matrix()[5], seed=0, null_draws=3)
    assert diverse.unique_records > duplicated.unique_records
    assert diverse.target_tokens == duplicated.target_tokens


def test_gate_fails_closed_without_durable_checkpoints() -> None:
    cells = tuple(
        run_regime_cell(spec, seed=seed, null_draws=3)
        for spec in build_fixture_matrix()
        for seed in (0, 1, 2)
    )
    gate = decide_regime_gate(cells, semantic_floor_verdict="inconclusive")
    assert gate.schema == "SpectralRegimeGateV1"
    assert gate.verdict == "inconclusive"
    assert gate.eligible_shapes == ("16x16",)
    assert gate.outcome_relationship["n"] == 18.0
    assert "spectral_lr_control" in gate.blocked_claims
    assert any("no durable" in reason for reason in gate.rationale)


def test_markdown_states_honest_scope() -> None:
    from slm_training.harnesses.experiments.slm216_spectral_regime import (
        run_spectral_regime_matrix,
    )

    repo_root = Path(__file__).resolve().parents[3]
    report = run_spectral_regime_matrix(repo_root=repo_root, seeds=(0,), null_draws=3)
    markdown = render_markdown(report)
    assert "No reusable checkpoint" in markdown
    assert "`inconclusive`" in markdown
