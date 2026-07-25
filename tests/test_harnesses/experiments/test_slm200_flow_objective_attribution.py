"""SLM-200 matched objective-attribution contracts."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from slm_training.harnesses.experiments.slm200_flow_objective_attribution import (
    OBJECTIVES,
    _shuffle_targets,
    run_matrix,
)
from slm_training.flow.targets import from_bridge_rows
from slm_training.data.flow.bridge_corpus import load_corpus


def test_objective_registry_has_every_preregistered_production_arm() -> None:
    assert [spec.arm_id for spec in OBJECTIVES] == [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
    ]
    assert next(spec for spec in OBJECTIVES if spec.arm_id == "A8").shuffled_targets
    assert next(spec for spec in OBJECTIVES if spec.arm_id == "A9").exact_only
    assert len({json.dumps(asdict(spec), sort_keys=True) for spec in OBJECTIVES}) == 9


def test_shuffled_control_preserves_unknown_as_unsupervised() -> None:
    rows, _, _ = load_corpus(
        Path("src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture")
    )
    train_rows = tuple(row for row in rows if row.split == "train")
    original = from_bridge_rows(train_rows)
    shuffled = _shuffle_targets(original)
    for row, target in zip(train_rows, shuffled, strict=True):
        assert not (
            set(row.unknown_candidate_ids) & set(target.supervised_candidate_ids)
        )
        assert target.fidelity == "surrogate_rate_weight"


def test_fixture_matrix_is_matched_and_refuses_a_flow_claim() -> None:
    report = run_matrix(seeds=(0, 1), steps=2, max_wall_minutes=0.5)
    assert set(report["arms"]) == {f"A{index}" for index in range(10)}
    assert report["arms"]["A0"]["status"] == "unavailable"
    assert report["arms"]["A9"]["oracle"]["rate_fit"]["max_abs_error"] < 1e-3
    assert report["primary_parity"]["declared_differences_only"]
    assert report["confirmation"]["status"] == "not_touched"
    assert report["confirmation"]["touch_ledger"] == []
    assert report["analysis"]["flow_win"] is False
    assert report["honest_verdict"] == "no_conclusion_underpowered_fixture"
    assert report["checkpoint"]["written"] is False


@pytest.mark.parametrize("value", [0.0, -1.0, 3.01])
def test_hard_wall_cap_is_enforced(value: float) -> None:
    with pytest.raises(ValueError):
        run_matrix(seeds=(0,), steps=1, max_wall_minutes=value)
