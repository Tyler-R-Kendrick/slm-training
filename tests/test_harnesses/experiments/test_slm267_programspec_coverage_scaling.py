"""Tests for SLM-267 (VSD2-02) ProgramSpec coverage-scaling wiring harness."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.slm267_programspec_coverage_scaling import (
    DEFAULT_CONFIG,
    run_coverage_scaling_campaign,
    run_policy_arm,
    shard_seed,
)


def test_shard_seed_is_deterministic() -> None:
    assert shard_seed(0, 0, 0) == shard_seed(0, 0, 0)


def test_shard_seed_is_sensitive_to_every_input() -> None:
    base = shard_seed(0, 0, 0)
    assert shard_seed(0, 1, 0) != base
    assert shard_seed(0, 0, 1) != base
    assert shard_seed(1, 0, 0) != base


def test_shard_seed_rejects_negative_ids() -> None:
    with pytest.raises(ValueError):
        shard_seed(0, -1, 0)
    with pytest.raises(ValueError):
        shard_seed(0, 0, -1)


def test_run_policy_arm_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError):
        run_policy_arm("random", config=DEFAULT_CONFIG, seed=0, target_count=1)


def test_run_policy_arm_rejects_negative_target_count() -> None:
    with pytest.raises(ValueError):
        run_policy_arm("uniform", config=DEFAULT_CONFIG, seed=0, target_count=-1)


def test_run_policy_arm_is_deterministic() -> None:
    first = run_policy_arm(
        "coverage_targeted", config=DEFAULT_CONFIG, seed=7, target_count=40
    )
    second = run_policy_arm(
        "coverage_targeted", config=DEFAULT_CONFIG, seed=7, target_count=40
    )
    assert first == second


def test_uniform_arm_actually_ignores_coverage_score() -> None:
    """A uniform arm and a coverage-targeted arm from the same seed must diverge
    in acceptance order even though both draw from the identical candidate grid."""
    uniform = run_policy_arm(
        "uniform", config=DEFAULT_CONFIG, seed=0, target_count=40
    )
    targeted = run_policy_arm(
        "coverage_targeted", config=DEFAULT_CONFIG, seed=0, target_count=40
    )
    assert uniform.programs_to_full_coverage != targeted.programs_to_full_coverage


def test_campaign_rejects_non_positive_shards() -> None:
    with pytest.raises(ValueError):
        run_coverage_scaling_campaign(target_count=10, shards=0)


def test_campaign_is_deterministic_and_self_hashed() -> None:
    first = run_coverage_scaling_campaign(target_count=60, seed=1, shards=2)
    second = run_coverage_scaling_campaign(target_count=60, seed=1, shards=2)
    assert first == second
    assert first["manifest_hash"]


def test_campaign_coverage_targeted_reaches_full_coverage_no_slower() -> None:
    report = run_coverage_scaling_campaign(target_count=80, seed=0, shards=2)
    comparison = report["comparison"]
    assert comparison["uniform_programs_to_full_coverage"] is not None
    assert comparison["coverage_targeted_programs_to_full_coverage"] is not None
    assert (
        comparison["coverage_targeted_programs_to_full_coverage"]
        <= comparison["uniform_programs_to_full_coverage"]
    )
    assert comparison["coverage_targeted_reaches_full_coverage_no_slower"] is True


def test_campaign_is_honest_wiring_only() -> None:
    report = run_coverage_scaling_campaign(target_count=60, seed=0, shards=1)
    assert report["claim_class"] == "wiring"
    assert report["status"] == "inconclusive"
    assert report["limitations"]


def test_campaign_dedups_canonical_roots_in_memory() -> None:
    report = run_coverage_scaling_campaign(target_count=60, seed=0, shards=1)
    for shard_results in report["arms"].values():
        for result in shard_results:
            assert result["unique_canonical_roots"] <= result["accepted"]
            assert (
                result["unique_canonical_roots"] + result["duplicate_roots"]
                == result["accepted"]
            )
