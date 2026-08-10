"""M2: legal-edit hazard (flow) ranking lever over the compiler-decision seam.

The head only reorders already-legal candidates (invariant I5/I6): bias rows
cover exactly the supplied candidate ids, the tanh clamp caps its authority at
the configured weight, defective output degrades to identity order with a
counter, and the arm's zero-weight matched control prebuilds the same head.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = (
    'root = Stack([b1], "column")\n'
    "b1 = Card([b2, b3])\n"
    'b2 = TextContent(":slot_0")\n'
    'b3 = TextContent(":slot_1")'
)


def _records() -> list[ExampleRecord]:
    return [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui=HERO,
            placeholders=[":slot_0", ":slot_1"],
            split="train",
        )
    ]


def _model(**kwargs) -> TwoTowerModel:
    torch.manual_seed(123)
    return TwoTowerModel.from_records(
        _records(),
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            output_tokenizer="lexer",
            compiler_decode_mode="tree",
            **kwargs,
        ),
    )


def test_legal_edit_hazard_profile_prebuild_parity() -> None:
    control = _model(structural_aux_head_profile="legal-edit-hazard")
    candidate = _model(
        structural_aux_head_profile="legal-edit-hazard",
        legal_edit_hazard_loss_weight=1.0,
        legal_edit_hazard_decode_weight=1.0,
    )
    assert control.legal_edit_hazard_head is not None
    assert candidate.legal_edit_hazard_head is not None
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )


def test_hazard_bias_covers_only_supplied_candidates_and_is_bounded() -> None:
    model = _model(
        structural_aux_head_profile="legal-edit-hazard",
        legal_edit_hazard_loss_weight=1.0,
        legal_edit_hazard_decode_weight=0.5,
    )
    ctx = torch.randn(1, 4, 32)
    ctx_pad = torch.zeros(1, 4, dtype=torch.bool)
    candidates = (2, 3, 5)
    bias = model._legal_edit_hazard_bias(ctx, ctx_pad, torch.randn(32), candidates)
    assert bias is not None
    assert bias.shape == (len(candidates),)
    assert bool((bias.abs() <= 0.5 + 1e-6).all())


def test_hazard_bias_disabled_returns_none() -> None:
    model = _model(structural_aux_head_profile="legal-edit-hazard")
    ctx = torch.randn(1, 4, 32)
    ctx_pad = torch.zeros(1, 4, dtype=torch.bool)
    assert (
        model._legal_edit_hazard_bias(ctx, ctx_pad, torch.randn(32), (2, 3)) is None
    )


def test_hazard_bias_non_finite_falls_back_with_counter() -> None:
    model = _model(
        structural_aux_head_profile="legal-edit-hazard",
        legal_edit_hazard_loss_weight=1.0,
        legal_edit_hazard_decode_weight=1.0,
    )

    class _InfHead(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(features.shape[0], 1)
            out[0, 0] = float("inf")
            return out

    model.legal_edit_hazard_head = _InfHead()
    ctx = torch.randn(1, 4, 32)
    ctx_pad = torch.zeros(1, 4, dtype=torch.bool)
    with collect_decode_stats() as stats:
        bias = model._legal_edit_hazard_bias(ctx, ctx_pad, torch.randn(32), (2, 3))
    assert bias is None
    assert stats.legal_edit_hazard_fallbacks == 1


def test_hazard_decode_weight_requires_training_loss() -> None:
    with pytest.raises(ValueError, match="legal_edit_hazard_decode_weight"):
        TwoTowerConfig(
            output_tokenizer="lexer",
            compiler_decode_mode="tree",
            legal_edit_hazard_decode_weight=1.0,
        )


def test_hazard_training_loss_decreases() -> None:
    model = _model(
        structural_aux_head_profile="legal-edit-hazard",
        legal_edit_hazard_loss_weight=1.0,
        legal_edit_hazard_decode_weight=1.0,
    )
    records = _records()
    optimizer = torch.optim.Adam(
        model.legal_edit_hazard_head.parameters(), lr=0.05
    )
    losses: list[float] = []
    for _ in range(5):
        model.training_loss(records)
        auxiliary = model.take_detached_auxiliary_loss()
        assert auxiliary is not None
        optimizer.zero_grad()
        auxiliary.backward()
        optimizer.step()
        losses.append(float(auxiliary))
    assert losses[-1] < losses[0]


def test_hazard_loss_reports_seam_metrics() -> None:
    model = _model(
        structural_aux_head_profile="legal-edit-hazard",
        legal_edit_hazard_loss_weight=1.0,
        legal_edit_hazard_decode_weight=1.0,
    )
    model.training_loss(_records())
    metrics = model.last_training_metrics
    assert metrics["legal_edit_hazard_decision_rows"] > 0
    assert metrics["legal_edit_hazard_loss"] >= metrics["legal_edit_hazard_mass_loss"]
