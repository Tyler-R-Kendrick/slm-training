from __future__ import annotations

import pytest

from slm_training.evals.cap2_operator import (
    CAP2_DISPOSITION_SCORE_SCHEMA,
    score_disposition_predictions,
)
from slm_training.levers import TURN_DISPOSITION_WRONG_EMIT_PENALTY_RATIO


def test_wrong_op_rate_and_abstention_rate_are_reported_separately() -> None:
    rows = [
        {"predicted_disposition": "emit", "emit_correct": True},
        {"predicted_disposition": "emit", "emit_correct": False},
        {"predicted_disposition": "clarify"},
        {"predicted_disposition": "clarify"},
        {"predicted_disposition": "answer"},
        {"predicted_disposition": "out_of_scope"},
        {"predicted_disposition": "emit", "emit_correct": False},
        {"predicted_disposition": "emit", "emit_correct": True},
    ]
    score = score_disposition_predictions(rows)
    assert score["schema"] == CAP2_DISPOSITION_SCORE_SCHEMA
    assert score["case_count"] == 8
    assert score["wrong_op_count"] == 2
    assert score["abstention_count"] == 2
    assert score["wrong_op_rate"] == pytest.approx(2 / 8)
    assert score["abstention_rate"] == pytest.approx(2 / 8)
    assert score["wrong_emit_penalty_ratio"] == TURN_DISPOSITION_WRONG_EMIT_PENALTY_RATIO


def test_composite_penalizes_a_wrong_emit_more_than_a_clarify() -> None:
    all_wrong_emit = [{"predicted_disposition": "emit", "emit_correct": False}] * 4
    all_clarify = [{"predicted_disposition": "clarify"}] * 4
    wrong_emit_score = score_disposition_predictions(all_wrong_emit)
    clarify_score = score_disposition_predictions(all_clarify)
    assert wrong_emit_score["wrong_op_rate"] == 1.0
    assert clarify_score["abstention_rate"] == 1.0
    assert (
        wrong_emit_score["composite_penalized_error_rate"]
        > clarify_score["composite_penalized_error_rate"]
    )
    # The exact registered penalty ratio determines the asymmetry precisely.
    ratio = TURN_DISPOSITION_WRONG_EMIT_PENALTY_RATIO
    assert wrong_emit_score["composite_penalized_error_rate"] == pytest.approx(
        ratio / (ratio + 1.0)
    )
    assert clarify_score["composite_penalized_error_rate"] == pytest.approx(
        1.0 / (ratio + 1.0)
    )


def test_emit_rows_require_emit_correct() -> None:
    with pytest.raises(ValueError, match="emit_correct"):
        score_disposition_predictions([{"predicted_disposition": "emit"}])


def test_empty_rows_and_bad_ratio_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        score_disposition_predictions([])
    with pytest.raises(ValueError, match="positive"):
        score_disposition_predictions(
            [{"predicted_disposition": "clarify"}], wrong_emit_penalty_ratio=0.0
        )
