"""SLM-406 temporal feature-row leakage and split regression coverage."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.operator_failure_prediction import (
    assert_group_disjoint_splits,
    build_operator_failure_feature_rows,
)


def _details(*, trace: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "id": "request-a",
            "decode_outcome": "model_invalid",
            "temporal_decode_evidence": [trace],
        }
    ]


def test_feature_rows_keep_terminal_outcome_as_label_not_feature() -> None:
    rows = build_operator_failure_feature_rows(
        _details(trace={"position": 3, "legal_candidates": 2, "forced": False}),
        checkpoint_sha="checkpoint-a",
        canvas_cap=8,
        target_cluster_by_id={"request-a": "cluster-a"},
    )

    assert len(rows) == 1
    assert rows[0].failure
    assert rows[0].group_id == "cluster-a:checkpoint-a"
    assert rows[0].features == {
        "normalized_position": 0.5,
        "legal_candidates": 2.0,
        "forced": 0.0,
    }
    assert "decode_outcome" not in rows[0].features


def test_feature_rows_reject_terminal_outcome_leakage() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        build_operator_failure_feature_rows(
            _details(
                trace={
                    "position": 0,
                    "legal_candidates": 1,
                    "parse_ok": False,
                }
            ),
            checkpoint_sha="checkpoint-a",
            canvas_cap=8,
        )


def test_group_split_contract_requires_every_group_to_be_assigned() -> None:
    rows = build_operator_failure_feature_rows(
        _details(trace={"position": 0, "legal_candidates": 1}),
        checkpoint_sha="checkpoint-a",
        canvas_cap=8,
    )

    with pytest.raises(ValueError, match="requires"):
        assert_group_disjoint_splits(rows, {})
    assert_group_disjoint_splits(rows, {"request-a:checkpoint-a": "test"})
