"""SLM-292 regression coverage for the default-off semantic contrast objective."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.data.contract import canonicalize_example_template_markers
from slm_training.models.semantic_contrast_loss import (
    load_semantic_contrast_pairs,
    pairwise_margin_loss,
    semantic_contrast_metadata,
)
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

POSITIVE = 'root = Stack([cta])\ncta = Button(":external.label")'
NEGATIVE = 'root = Stack([copy])\ncopy = TextContent(":external.label")'


def _pair_row(*, negative_role: str = "negative") -> dict:
    positive = ExampleRecord(
        "positive",
        "Build a CTA",
        POSITIVE,
        split="train",
        placeholders=[":external.label"],
    )
    negative = ExampleRecord(
        "negative",
        "Build a CTA",
        NEGATIVE,
        split="train",
        placeholders=[":external.label"],
    )
    source = "source-1"
    return {
        "pair_id": "pair-1",
        "admitted": True,
        "family": "binding",
        "transform_id": "binding_swap_symbol",
        "source_program_id": source,
        "positive": {"role": "positive", "record": positive.to_dict(), "source_program_id": source},
        "negative": {"role": negative_role, "record": negative.to_dict(), "source_program_id": source},
    }


def test_margin_zero_when_positive_score_clears_margin() -> None:
    result = pairwise_margin_loss(
        torch.tensor([3.0]), torch.tensor([1.0]), margin=1.0
    )
    assert float(result.loss) == 0.0
    assert float(result.score_distance) == 2.0


def test_margin_penalizes_inverted_pair() -> None:
    result = pairwise_margin_loss(
        torch.tensor([1.0]), torch.tensor([3.0]), margin=1.0
    )
    assert float(result.loss) == 3.0
    assert float(result.violation_rate) == 1.0


def test_pair_loader_rejects_malformed_roles(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(_pair_row(negative_role="positive")) + "\n")
    with pytest.raises(ValueError, match="pair roles"):
        load_semantic_contrast_pairs(path)


def test_pair_loader_marks_complete_pair_metadata(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(_pair_row()) + "\n")
    pair = load_semantic_contrast_pairs(path)[0]
    assert pair.positive.openui.endswith('Button(":slot_0")')
    assert pair.negative.openui.endswith('TextContent(":slot_0")')
    assert semantic_contrast_metadata(pair.positive) == {
        "pair_id": "pair-1",
        "family": "binding",
        "transform_id": "binding_swap_symbol",
        "source_program_id": "source-1",
        "role": "positive",
    }
    assert semantic_contrast_metadata(pair.negative)["role"] == "negative"


def test_pair_loader_preserves_shared_ordinals_when_negative_drops_slot(
    tmp_path: Path,
) -> None:
    row = _pair_row()
    row["positive"]["record"]["openui"] = (
        'root = Stack([a, b])\na = Button(":external.a")\n'
        'b = Button(":external.b")'
    )
    row["positive"]["record"]["placeholders"] = [":external.a", ":external.b"]
    row["positive"]["record"]["prompt"] = "Use :external.a and :external.b"
    row["negative"]["record"]["openui"] = (
        'root = Stack([a, b])\na = Button(":slot_1.label")\n'
        'b = Button(":slot_1.label")'
    )
    row["negative"]["record"]["placeholders"] = [":slot_1.label"]
    row["negative"]["record"]["prompt"] = "Use :external.a and :external.b"

    pair = load_semantic_contrast_pairs(_write_pair(tmp_path=tmp_path, row=row))[0]

    assert pair.positive.placeholders == [":slot_0", ":slot_1"]
    assert pair.negative.placeholders == [":slot_0", ":slot_1"]
    assert pair.negative.openui.count(":slot_1") == 2
    assert ":slot_0" not in pair.negative.openui


def _model(records: list[ExampleRecord], **kwargs) -> TwoTowerModel:
    return TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            output_tokenizer="lexer",
            **kwargs,
        ),
    )


def test_zero_weight_control_matches_candidate_reconstruction_mask(
    tmp_path: Path,
) -> None:
    raw = _pair_row()
    base = canonicalize_example_template_markers(
        ExampleRecord.from_dict(raw["positive"]["record"]),
        marker_order=[":external.label"],
    )
    pair = load_semantic_contrast_pairs(_write_pair(tmp_path=tmp_path, row=raw))[0]
    batch = [base, pair.positive, pair.negative]
    torch.manual_seed(17)
    control = _model(batch, semantic_contrast_loss_weight=0.0)
    torch.manual_seed(17)
    candidate = _model(
        batch,
        semantic_contrast_loss_weight=0.5,
        semantic_contrast_margin=1.0,
    )
    rng_state = control._rng.getstate()
    torch.manual_seed(41)
    control_loss = control.training_loss(batch)
    candidate._rng.setstate(rng_state)
    torch.manual_seed(41)
    candidate_loss = candidate.training_loss(batch)

    control_metrics = control.last_training_metrics
    candidate_metrics = candidate.last_training_metrics
    assert control_metrics["primary_final_reconstruction_loss"] == pytest.approx(
        candidate_metrics["primary_final_reconstruction_loss"]
    )
    for key in (
        "semantic_contrast_ce_negative_rows_excluded",
        "semantic_contrast_ce_scored_tokens",
        "semantic_contrast_ce_mask_sha256",
    ):
        assert control_metrics[key] == candidate_metrics[key]
    assert control_metrics["semantic_contrast_ce_negative_rows_excluded"] == 1
    assert control_metrics["semantic_contrast_loss"] == 0.0
    assert candidate_metrics["semantic_contrast_pairs"] == 1
    assert float(candidate_loss - control_loss) == pytest.approx(
        0.5 * candidate_metrics["semantic_contrast_loss"], abs=2e-6
    )


def _write_pair(*, tmp_path: Path, row: dict) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


def test_enabled_objective_logs_complete_pair_metrics(tmp_path: Path) -> None:
    pair = load_semantic_contrast_pairs(_write_pair(tmp_path=tmp_path, row=_pair_row()))[0]
    model = _model(
        [pair.positive, pair.negative],
        semantic_contrast_loss_weight=0.5,
        semantic_contrast_margin=1.0,
    )
    loss = model.training_loss([pair.positive, pair.negative])
    assert torch.isfinite(loss)
    metrics = model.last_training_metrics
    assert metrics["semantic_contrast_loss_weight"] == 0.5
    assert metrics["semantic_contrast_pairs"] == 1
    assert metrics["semantic_contrast_family_sides"] == {"binding": 2}
    assert "semantic_contrast_positive_nll" in metrics
    assert "semantic_contrast_negative_nll" in metrics
