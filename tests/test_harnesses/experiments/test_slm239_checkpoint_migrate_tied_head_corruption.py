"""Tests for the SLM-239 checkpoint-migrate output-head corruption harness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.experiments.slm239_checkpoint_migrate_tied_head_corruption import (
    EXPERIMENT_ID,
    MATRIX_SET,
    CheckpointMigrateCorruptionReport,
    render_markdown,
    run_checkpoint_migrate_corruption_sweep,
)


def test_sweep_runs_both_tie_arms_for_every_seed() -> None:
    report = run_checkpoint_migrate_corruption_sweep(seeds=(0, 1))

    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.claim_class == "wiring"
    assert report.status == "fixture"
    assert len(report.results) == 4  # 2 seeds x 2 tie arms

    for result in report.results:
        assert result.vocab_size_matched
        assert result.old_vocab_size == result.new_vocab_size
        assert result.shifted_token_count > 0
        assert result.tok_weight_correct_fraction is not None
        assert result.lm_head_present

    assert report.version_stamp.get("stamp_schema") == "version_stamp/v1"


def test_tied_arm_clobbers_the_shared_embedding_remap() -> None:
    # This is the specific mechanism claim: with tie_output_embedding=True,
    # denoiser.tok.weight and denoiser.lm_head.weight alias the same storage,
    # and migrate_twotower_checkpoint processes lm_head's naive whole-tensor
    # copy after tok's correct per-token remap, so the correct remap gets
    # wholesale overwritten by the raw, un-remapped old matrix.
    report = run_checkpoint_migrate_corruption_sweep(seeds=(0,))
    tied = [r for r in report.results if r.tie_output_embedding][0]
    assert tied.tok_weight_correct_fraction == 0.0
    assert tied.lm_head_correct_fraction == 0.0
    assert tied.tok_weight_whole_matches_raw_old is True
    assert tied.corrupted is True


def test_untied_arm_leaves_tok_correct_but_lm_head_stale() -> None:
    # With tie_output_embedding=False the two tensors are independent:
    # tok.weight is correctly remapped (untouched by the lm_head branch),
    # but lm_head.weight is never remapped by token string at all.
    report = run_checkpoint_migrate_corruption_sweep(seeds=(0,))
    untied = [r for r in report.results if not r.tie_output_embedding][0]
    assert untied.tok_weight_correct_fraction == 1.0
    assert untied.lm_head_correct_fraction == 0.0
    assert untied.tok_weight_whole_matches_raw_old is False
    assert untied.corrupted is True


def test_report_roundtrips_through_dict() -> None:
    report = run_checkpoint_migrate_corruption_sweep(seeds=(0,))
    payload = report.to_dict()
    restored = CheckpointMigrateCorruptionReport.from_dict(payload)
    assert restored.to_dict() == payload


def test_render_markdown_includes_disposition_and_table() -> None:
    report = run_checkpoint_migrate_corruption_sweep(seeds=(0, 1))
    text = render_markdown(report)
    assert report.disposition in text
    assert (
        "| tie | seed | old vocab | new vocab | shifted tokens | tok correct frac | "
        "lm_head correct frac | tok==raw old (whole) | corrupted |" in text
    )
    assert "No-go for trusting migrate_twotower_checkpoint" in text
