from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from slm_training.data.flow.bridge_corpus import load_corpus
from slm_training.flow.targets import from_bridge_rows
from slm_training.models.legal_edit_batch import LegalEditBatch
from slm_training.models.legal_edit_flow import (
    LegalEditFlow,
    LegalEditFlowConfig,
    legal_edit_flow_losses,
)
from slm_training.models.legal_edit_scorer import DirectLegalEditPolicy, LegalEditScorer

CORPUS = Path("src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture")


def _fixture() -> tuple[LegalEditBatch, tuple]:
    rows, candidate_sets, _ = load_corpus(CORPUS)
    return LegalEditBatch.pack(rows, candidate_sets), from_bridge_rows(rows)


def test_default_off_preserves_existing_path() -> None:
    model = LegalEditFlow()
    batch, _ = _fixture()
    with pytest.raises(RuntimeError, match="disabled"):
        model(batch)
    with pytest.raises(ValueError, match="min_rate"):
        LegalEditFlowConfig(enabled=True, min_rate=float("nan"))
    with pytest.raises(ValueError, match="min_rate"):
        LegalEditFlowConfig(enabled=True, min_rate=-1.0)


def test_nonnegative_rates_positive_hazard_and_separate_losses() -> None:
    batch, targets = _fixture()
    model = LegalEditFlow(LegalEditFlowConfig(enabled=True))
    prediction = model(
        batch, schedule_progress=torch.zeros(len(batch.row_ids))
    )
    total, losses = legal_edit_flow_losses(prediction, batch, targets)
    assert bool((prediction.edge_rates >= 0).all())
    assert bool((prediction.row_hazards > 0).all())
    assert set(losses) == {
        "edge_rate",
        "total_hazard",
        "multi_positive_mass",
        "terminal_absorption",
    }
    assert torch.isfinite(total)
    swapped = (replace(targets[0], row_id="wrong"),) + targets[1:]
    with pytest.raises(ValueError, match="row identity"):
        legal_edit_flow_losses(prediction, batch, swapped)


def test_checkpoint_round_trip_and_direct_migration_default_off(tmp_path: Path) -> None:
    model = LegalEditFlow(LegalEditFlowConfig(enabled=True))
    path = tmp_path / "flow.pt"
    model.save(path)
    loaded = LegalEditFlow.from_checkpoint(path)
    assert loaded.config.enabled
    direct = DirectLegalEditPolicy(LegalEditScorer())
    direct_path = tmp_path / "direct.pt"
    direct.save(direct_path)
    migrated = LegalEditFlow.from_checkpoint(direct_path)
    assert not migrated.config.enabled
