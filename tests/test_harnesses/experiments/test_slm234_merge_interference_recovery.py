"""Tests for the SLM-234 (CKM0-01) TIES-vs-average merge signal recovery
probe."""

from __future__ import annotations

import torch

from slm_training.harnesses.experiments.slm234_merge_interference_recovery import (
    EXPERIMENT_ID,
    MATRIX_SET,
    Slm234Report,
    _build_ground_truth,
    _build_sibling_delta,
    _pair_rows,
    _score_merged_delta,
    render_markdown,
    run_merge_interference_matrix,
)


def test_build_ground_truth_is_deterministic_and_has_signal_and_noise() -> None:
    parent_a, mask_a, sign_a = _build_ground_truth(seed=0)
    parent_b, mask_b, sign_b = _build_ground_truth(seed=0)
    for name in parent_a:
        assert torch.equal(parent_a[name], parent_b[name])
        assert torch.equal(mask_a[name], mask_b[name])
        assert torch.equal(sign_a[name], sign_b[name])
        # Some signal coordinates and some noise coordinates exist.
        assert 0 < int(mask_a[name].sum()) < mask_a[name].numel()
        # true_sign is nonzero exactly where signal_mask is set.
        assert torch.equal(sign_a[name] != 0, mask_a[name])


def test_build_ground_truth_differs_across_seeds() -> None:
    parent_a, mask_a, _ = _build_ground_truth(seed=0)
    parent_b, mask_b, _ = _build_ground_truth(seed=1)
    differs = any(
        not torch.equal(parent_a[name], parent_b[name]) or not torch.equal(mask_a[name], mask_b[name])
        for name in parent_a
    )
    assert differs


def test_sibling_delta_with_zero_conflict_matches_true_sign() -> None:
    parent, mask, sign = _build_ground_truth(seed=0)
    for name, tensor in parent.items():
        shape = tuple(tensor.shape)
        delta = _build_sibling_delta(shape, mask[name], sign[name], 0.0, sibling_seed=42)
        signal = mask[name].bool()
        # Zero conflict probability -> every signal coordinate keeps true sign.
        assert torch.equal(torch.sign(delta[signal]), sign[name][signal])


def test_score_merged_delta_perfect_recovery_gives_cosine_one() -> None:
    _, mask, sign = _build_ground_truth(seed=0)
    # A merged delta identical to the ground-truth sign vector should score
    # a perfect cosine similarity and full magnitude/sign recovery.
    merged = {name: sign[name].clone() for name in sign}
    scores = _score_merged_delta(merged, mask, sign)
    assert scores["cosine_similarity"] > 0.999
    assert scores["signal_sign_recovery_rate"] == 1.0
    assert scores["signal_magnitude_recovery"] > 0.999


def test_run_merge_interference_matrix_shape() -> None:
    report = run_merge_interference_matrix(seeds=(0, 1), conflict_probs=(0.0, 0.3))
    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    # 2 seeds x 2 conflict_probs x 2 methods = 8 rows.
    assert len(report.rows) == 8
    assert len(report.metric_summaries) == 4
    assert report.disposition in {
        "fully_confirmed",
        "partial_confirmation_mechanism_specific",
        "no_advantage_detected",
    }


def test_noise_residual_is_always_at_least_as_good_for_ties() -> None:
    # merge_checkpoints(method="ties") trims small-magnitude entries per
    # sibling before merging; on this fixture's noise coordinates (small,
    # independent, random sign) that should never do worse than naive
    # averaging.
    report = run_merge_interference_matrix(seeds=(0, 1, 2), conflict_probs=(0.0, 0.15, 0.3, 0.45))
    paired = _pair_rows(list(report.rows))
    for (_seed, _level), pair in paired.items():
        avg, ties = pair["average"], pair["ties"]
        assert ties.mean_abs_noise_residual <= avg.mean_abs_noise_residual + 1e-6


def test_gate_hash_free_report_roundtrips_through_dict() -> None:
    report = run_merge_interference_matrix(seeds=(0,), conflict_probs=(0.0, 0.3))
    payload = report.to_dict()
    restored = Slm234Report.from_dict(payload)
    assert restored.to_dict() == payload


def test_run_is_deterministic_given_same_seeds() -> None:
    a = run_merge_interference_matrix(seeds=(0, 1), conflict_probs=(0.0, 0.3))
    b = run_merge_interference_matrix(seeds=(0, 1), conflict_probs=(0.0, 0.3))
    a_rows = sorted(r.to_dict().items() for r in a.rows)
    b_rows = sorted(r.to_dict().items() for r in b.rows)
    assert a_rows == b_rows
    assert a.disposition == b.disposition


def test_render_markdown_includes_disposition_and_tables() -> None:
    report = run_merge_interference_matrix(seeds=(0,), conflict_probs=(0.0, 0.3))
    text = render_markdown(report)
    assert report.disposition in text
    assert "SLM-234" in text
    assert "| metric | higher is better |" in text
    assert "No-go for promotion" in text
