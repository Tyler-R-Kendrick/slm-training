"""Tests for SLM-230 recurrence observability contracts."""

from __future__ import annotations

import math

import pytest
import torch

from slm_training.harnesses.experiments.slm230_recurrence_observability import (
    EXIT_POLICY_SCHEMA,
    ExitMode,
    RecurrenceExitPolicyV1,
    RecurrenceVerdict,
    classify_recurrence,
    distribution_metrics,
    histogram_matched_control,
    select_exit_depth,
)


def test_identical_distributions_have_zero_kl_and_js() -> None:
    logits = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    metrics = distribution_metrics(
        logits,
        mask=torch.tensor([True, True]),
        previous_logits=logits.clone(),
        legal_candidate_ids=torch.tensor([0, 1]),
        top_k=2,
    )
    assert metrics["full_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["full_js"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["legal_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["legal_js"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["top1_stable"] is True


def test_legal_candidate_identity_must_be_frozen_and_unique() -> None:
    logits = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="nonempty unique"):
        distribution_metrics(
            logits,
            mask=torch.tensor([True]),
            previous_logits=logits,
            legal_candidate_ids=torch.tensor([0, 0]),
            top_k=1,
        )


def test_production_like_policy_rejects_oracle_signals() -> None:
    policy = RecurrenceExitPolicyV1(
        mode=ExitMode.KL_PLATEAU,
        minimum_depth=1,
        maximum_depth=4,
        kl_threshold=0.1,
        allowed_signals=("full_kl_from_previous", "reward_score"),
        calibration_split_hash="abc",
    )
    with pytest.raises(ValueError, match="cannot consume"):
        policy.validate()


def test_adaptive_policy_requires_frozen_calibration_split() -> None:
    policy = RecurrenceExitPolicyV1(
        mode=ExitMode.TOPK_STABLE,
        minimum_depth=1,
        maximum_depth=4,
        topk_stability_k=5,
        allowed_signals=("topk_stable",),
    )
    with pytest.raises(ValueError, match="calibration split"):
        policy.validate()


def test_kl_exit_uses_only_consecutive_plateau_rows() -> None:
    policy = RecurrenceExitPolicyV1(
        mode=ExitMode.KL_PLATEAU,
        minimum_depth=1,
        maximum_depth=4,
        consecutive_depths=2,
        kl_threshold=0.1,
        fallback_depth=4,
        allowed_signals=("full_kl_from_previous",),
        calibration_split_hash="abc",
    )
    observations = [
        {"evaluated_depth": 1, "full_kl_from_previous": None},
        {"evaluated_depth": 2, "full_kl_from_previous": 0.08},
        {"evaluated_depth": 3, "full_kl_from_previous": 0.09},
        {"evaluated_depth": 4, "full_kl_from_previous": 0.20},
    ]
    assert select_exit_depth(observations, policy) == 3


def test_histogram_control_preserves_depth_counts() -> None:
    selected = [2, 2, 4, 3]
    control = histogram_matched_control(
        selected,
        record_ids=["d", "b", "a", "c"],
    )
    assert sorted(control) == ["a", "b", "c", "d"]
    assert sorted(control.values()) == sorted(selected)


def test_regressing_free_running_curve_cannot_be_refining() -> None:
    rows = []
    for record_id, rewards in (("a", (0.2, 0.3, 0.1)), ("b", (0.1, 0.2, 0.1))):
        for depth, reward in enumerate(rewards, start=1):
            rows.append(
                {
                    "record_id": record_id,
                    "evaluated_depth": depth,
                    "reward_score": reward,
                    "full_kl_from_previous": None if depth == 1 else 0.1 / depth,
                    "numerical_status": "finite",
                }
            )
    verdict = classify_recurrence(
        rows,
        heldout_record_ids=["a", "b"],
        early_exit_qualified=True,
    )
    assert verdict is not RecurrenceVerdict.REFINING
    assert verdict is RecurrenceVerdict.OSCILLATORY


def test_contract_schema_is_stable() -> None:
    policy = RecurrenceExitPolicyV1(
        mode=ExitMode.FIXED,
        minimum_depth=1,
        maximum_depth=2,
    )
    payload = policy.to_dict()
    assert payload["schema"] == EXIT_POLICY_SCHEMA
    assert payload["mode"] == "fixed"
    assert math.isfinite(float(payload["maximum_depth"]))
