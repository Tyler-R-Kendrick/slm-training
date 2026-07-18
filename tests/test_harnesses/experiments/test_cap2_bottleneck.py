"""Regression tests for the CAP2-01 K-ary bottleneck matrix harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch")

from slm_training.harnesses.experiments.cap2_bottleneck import (
    BottleneckArm,
    build_matrix,
    evaluate_arm,
    evaluate_injective_arm,
    evaluate_robust_arm,
    fixture_states,
    load_state_report,
    run_matrix,
)


M = 41


def test_build_matrix_default_has_boundary_and_robust_arms() -> None:
    arms = build_matrix(M)
    ids = {a.arm_id for a in arms}
    assert "b2d5" in ids
    assert "b2d6" in ids
    assert "t3d3" in ids
    assert "t3d4" in ids
    assert "k2d6" in ids
    assert "k4d3" in ids
    assert "k8d2" in ids
    assert "k7d4_robust" in ids
    assert "k3d7_robust" in ids
    assert "direct_one_hot" in ids


def test_below_capacity_arm_fails_for_41_states() -> None:
    states, _ = fixture_states(M)
    arm = BottleneckArm("b2d5", 2, 5, M, mode="injective", seed=0)
    result = evaluate_injective_arm(arm, states)
    assert result.capacity < M
    assert result.exact_reconstruction_rate < 1.0
    assert result.collision_count > 0
    assert result.leakage is False


def test_above_capacity_arm_reconstructs_41_states() -> None:
    states, _ = fixture_states(M)
    arm = BottleneckArm("t3d4", 3, 4, M, mode="injective", seed=0)
    result = evaluate_injective_arm(arm, states)
    assert result.capacity >= M
    assert result.exact_reconstruction_rate == 1.0
    assert result.collision_count == 0
    assert result.occupied_codewords == M


def test_equal_capacity_arms_are_requirement_matched() -> None:
    states, _ = fixture_states(M)
    for arm_id, K, d in [("k2d6", 2, 6), ("k4d3", 4, 3), ("k8d2", 8, 2)]:
        arm = BottleneckArm(arm_id, K, d, M, mode="injective", seed=0)
        result = evaluate_injective_arm(arm, states)
        assert result.capacity == 64
        assert result.exact_reconstruction_rate == 1.0


def test_robust_mds_arm_corrects_one_substitution() -> None:
    states, _ = fixture_states(M)
    arm = BottleneckArm("k7d4_robust", 7, 4, M, mode="robust", corruption="one_substitution", seed=0)
    result = evaluate_robust_arm(arm, states)
    assert result.capacity >= M
    assert result.exact_reconstruction_rate == 1.0


def test_robust_ternary_arm_corrects_one_substitution() -> None:
    states, _ = fixture_states(M)
    arm = BottleneckArm("k3d7_robust", 3, 7, M, mode="robust", corruption="one_substitution", seed=0)
    result = evaluate_robust_arm(arm, states)
    assert result.capacity >= M
    assert result.exact_reconstruction_rate == 1.0


def test_learned_above_capacity_arm_reconstructs() -> None:
    states, _ = fixture_states(M)
    arm = BottleneckArm("learned_t3d4", 3, 4, M, mode="learned", train_steps=600, seed=0)
    result = evaluate_arm(arm, states)
    # Soft-trained MLP can converge to near-exact decoding without guaranteeing it.
    assert result.capacity >= M
    assert result.exact_reconstruction_rate >= 0.95
    assert result.leakage is False


def test_learned_below_capacity_arm_does_not_reconstruct() -> None:
    states, _ = fixture_states(M)
    arm = BottleneckArm("learned_b2d5", 2, 5, M, mode="learned", train_steps=300, seed=0)
    result = evaluate_arm(arm, states)
    assert result.capacity < M
    assert result.exact_reconstruction_rate < 1.0
    assert result.leakage is False


def test_run_matrix_produces_versioned_report(tmp_path: Path) -> None:
    report = run_matrix(M, seeds=(0,), arms_filter=("b2d5", "t3d4", "direct_one_hot"))
    assert report.state_count == M
    assert report.version == "cap2-02-v1"
    assert len(report.arms) == 3
    assert not any(a.leakage for a in report.arms)


def test_incomplete_state_report_rejected(tmp_path: Path) -> None:
    bad_report = tmp_path / "bad.json"
    bad_report.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="minimized_states"):
        load_state_report(bad_report)


def test_state_report_loading_uses_minimized_states(tmp_path: Path) -> None:
    report = {"minimized_states": 12, "nodes": []}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    data = load_state_report(path)
    assert data["minimized_states"] == 12


def test_matrix_includes_latent_codec_arms() -> None:
    arms = build_matrix(M)
    ids = {a.arm_id for a in arms}
    assert "fsq_2_3_3_4_5" in ids
    assert "lfq_d6" in ids
    assert "vq_64_d8" in ids
    assert "continuous_d6" in ids
    assert "uniform_b2d6" in ids


def test_fsq_arm_capacity_matches_radix_product() -> None:
    states, _ = fixture_states(M)
    arm = BottleneckArm(
        "fsq_2_3_3_4_5",
        0,
        0,
        M,
        mode="learned_codec",
        codec="fsq",
        radixes=(2, 3, 3, 4, 5),
        train_steps=1600,
        seed=0,
    )
    assert arm.capacity == 2 * 3 * 3 * 4 * 5
    result = evaluate_arm(arm, states)
    assert result.exact_reconstruction_rate == 1.0
    assert result.occupied_codewords == M


def test_lfq_arm_capacity_is_two_to_the_d() -> None:
    states, _ = fixture_states(M)
    arm = BottleneckArm("lfq_d6", 0, 6, M, mode="learned_codec", codec="lfq", train_steps=1600, seed=0)
    assert arm.capacity == 64
    result = evaluate_arm(arm, states)
    assert result.exact_reconstruction_rate == 1.0


def test_vq_arm_capacity_is_codebook_size() -> None:
    torch.manual_seed(0)
    states, _ = fixture_states(M)
    arm = BottleneckArm(
        "vq_64_d8",
        64,
        0,
        M,
        mode="learned_codec",
        codec="vq",
        latent_dim=8,
        train_steps=2400,
        seed=0,
    )
    assert arm.capacity == 64
    result = evaluate_arm(arm, states)
    assert result.exact_reconstruction_rate == 1.0


def test_continuous_arm_is_not_marked_as_leakage() -> None:
    states, _ = fixture_states(M)
    arm = BottleneckArm(
        "continuous_d6",
        0,
        0,
        M,
        mode="learned_codec",
        codec="continuous",
        latent_dim=6,
        train_steps=1200,
        seed=0,
    )
    result = evaluate_arm(arm, states)
    assert result.leakage is False
    assert "continuous" in result.notes[0]
