from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.data.flow.bridge_corpus import load_corpus
from slm_training.flow.reference.row import FlowTargetRowV1
from slm_training.flow.targets import (
    LegalEditRateTargetV1,
    from_bridge_rows,
    from_exact_rows,
)


def test_exact_target_preserves_rates_and_membership() -> None:
    row = FlowTargetRowV1(
        row_id="exact",
        time=0.5,
        exact_live_candidates=("next-a", "next-b"),
        target_rates={"next-a": 1.0, "next-b": 3.0},
        total_hazard=4.0,
    )
    target = from_exact_rows((row,))[0]
    assert target.fidelity == "exact_finite_graph"
    assert target.rate_dict() == row.target_rates
    assert target.total_hazard == sum(target.edge_rates)


def test_target_rejects_illegal_or_negative_rate() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        LegalEditRateTargetV1(
            row_id="bad",
            candidate_ids=("live",),
            edge_rates=(-1.0,),
            total_hazard=-1.0,
            positive_candidate_ids=(),
            supervised_candidate_ids=("live",),
            hazard_supervised=True,
            terminal_probability=0.0,
            time=0.0,
            fidelity="exact_finite_graph",
        )
    with pytest.raises(ValueError, match="rate keys"):
        from_exact_rows(
            (
                FlowTargetRowV1(
                    exact_live_candidates=("live",),
                    target_rates={"illegal": 1.0},
                    total_hazard=1.0,
                ),
            )
        )
    with pytest.raises(ValueError, match="nonnegative"):
        LegalEditRateTargetV1(
            row_id="nan",
            candidate_ids=("live",),
            edge_rates=(float("nan"),),
            total_hazard=float("nan"),
            positive_candidate_ids=("live",),
            supervised_candidate_ids=("live",),
            hazard_supervised=True,
            terminal_probability=0.0,
            time=0.0,
            fidelity="exact_finite_graph",
        )


def test_bridge_target_is_honestly_adapted_and_exact_membership() -> None:
    rows, _, _ = load_corpus(
        Path("src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture")
    )
    targets = from_bridge_rows(rows)
    assert all(item.fidelity == "adapted_path_approximation" for item in targets)
    assert all(item.total_hazard == 1.0 for item in targets)
    assert all(sum(item.edge_rates) == pytest.approx(1.0) for item in targets)
    assert all(
        item.candidate_ids == tuple(sorted(row.complete_candidate_ids))
        for item, row in zip(targets, rows, strict=True)
    )
    assert all(
        not (set(item.supervised_candidate_ids) & set(row.unknown_candidate_ids))
        for item, row in zip(targets, rows, strict=True)
    )
    assert all(not item.hazard_supervised for item in targets)
