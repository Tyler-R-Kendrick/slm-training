from __future__ import annotations

import pytest
import torch

from slm_training.models.turn_disposition_head import (
    TurnDispositionLabel,
    TurnDispositionReportV1,
    TurnDispositionTargetV1,
    turn_disposition_losses,
)


def test_forced_singleton_must_be_labeled_emit() -> None:
    with pytest.raises(ValueError, match="forced singleton"):
        TurnDispositionTargetV1(TurnDispositionLabel.CLARIFY, forced_singleton=True)
    target = TurnDispositionTargetV1(TurnDispositionLabel.EMIT, forced_singleton=True)
    assert target.label_index == 2


def test_mask_forced_zeroes_the_loss_without_touching_gradients() -> None:
    target = TurnDispositionTargetV1(TurnDispositionLabel.EMIT, forced_singleton=True)
    logits = torch.tensor([0.3, -0.2, 1.1], requires_grad=True)
    loss, values = turn_disposition_losses(logits, target, mask_forced=True)
    assert loss.item() == 0.0
    assert values["disposition"].item() == 0.0


def test_unforced_target_trains_a_real_cross_entropy_loss() -> None:
    target = TurnDispositionTargetV1(TurnDispositionLabel.CLARIFY, forced_singleton=False)
    logits = torch.tensor([0.0, 5.0, 0.0], requires_grad=True)
    loss, values = turn_disposition_losses(logits, target, mask_forced=True)
    # CLARIFY is index 1 and already dominant, so the loss should be small
    # but strictly positive (softmax is never exactly one-hot).
    assert 0.0 < loss.item() < 0.1
    assert torch.isfinite(values["disposition"])

    wrong_logits = torch.tensor([5.0, 0.0, 0.0], requires_grad=True)
    wrong_loss, _ = turn_disposition_losses(wrong_logits, target, mask_forced=True)
    assert wrong_loss.item() > loss.item()


def test_logits_must_match_the_three_way_label_space() -> None:
    target = TurnDispositionTargetV1(TurnDispositionLabel.EMIT, forced_singleton=False)
    with pytest.raises(ValueError, match="exactly 3"):
        turn_disposition_losses(torch.tensor([0.0, 0.0]), target)


def test_report_reports_wrong_emit_and_abstention_separately() -> None:
    predicted = [
        TurnDispositionLabel.EMIT,
        TurnDispositionLabel.EMIT,
        TurnDispositionLabel.CLARIFY,
        TurnDispositionLabel.ANSWER,
    ]
    expected = [
        TurnDispositionLabel.EMIT,
        TurnDispositionLabel.EMIT,
        TurnDispositionLabel.EMIT,
        TurnDispositionLabel.ANSWER,
    ]
    report = TurnDispositionReportV1.from_predictions(
        predicted=predicted,
        expected=expected,
        clarify_probabilities=(0.1, 0.2, 0.9, 0.05),
        clarify_targets=(0, 0, 1, 0),
        emit_correct=(True, False, False, False),
    )
    assert report.accuracy == pytest.approx(3 / 4)
    assert report.wrong_emit_rate == pytest.approx(1 / 4)
    assert report.abstention_rate == pytest.approx(1 / 4)
    assert 0.0 <= report.clarify_brier <= 1.0
    assert 0.0 <= report.clarify_ece <= 1.0
