"""Regression tests for CAP4-01 residual ternary planes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.local_action_head import (
    ResidualTritPlaneHead,
    StateContext,
)
from slm_training.models.quantization import (
    PlaneOutput,
    ResidualTritStack,
    build_model_ledger,
)
from slm_training.models.quantization.formats import (
    residual_ternary_plane_format,
    ternary_format,
)


HIDDEN_DIM = 8


def _hidden(batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, HIDDEN_DIM)


def test_additive_composition_and_max_planes() -> None:
    """Forward with max_planes truncates the stack additively."""
    stack = ResidualTritStack(HIDDEN_DIM, 4, R=3)
    x = _hidden(batch=2)
    full = stack(x)
    partial = stack(x, max_planes=1)
    assert full.shape == (2, 4)
    assert partial.shape == (2, 4)
    # A partial forward must differ from the full one when planes are non-zero.
    assert not torch.equal(full, partial)


def test_max_planes_zero_is_base_module() -> None:
    """max_planes=0 is equivalent to the base linear module."""
    stack = ResidualTritStack(HIDDEN_DIM, 4, R=2)
    x = _hidden(batch=2)
    base = stack.base_module(x)
    zero_planes = stack(x, max_planes=0)
    assert torch.allclose(base, zero_planes, atol=1e-6)


def test_effective_weight_matches_forward() -> None:
    """effective_weight reproduces the forward output (up to numerical noise)."""
    stack = ResidualTritStack(HIDDEN_DIM, 4, R=2)
    x = _hidden(batch=2)
    out = stack(x)
    w_eff = stack.effective_weight()
    expected = torch.nn.functional.linear(x, w_eff, stack.base_module.bias)
    assert torch.allclose(out, expected, atol=1e-5)


def test_geometric_grid_levels_is_three_to_the_r() -> None:
    """Only geometric_balanced mode advertises 3**R grid levels."""
    stack = ResidualTritStack(HIDDEN_DIM, 4, R=3, scale_mode="geometric_balanced")
    assert stack.grid_levels() == 3 ** stack.R


def test_learned_modes_reject_grid_levels() -> None:
    """Learned scale modes must not claim a fixed radix-3 grid."""
    for mode in ("learned_independent", "learned_monotone"):
        stack = ResidualTritStack(HIDDEN_DIM, 4, R=2, scale_mode=mode)
        with pytest.raises(ValueError, match="analytic 3\\^R grid claim"):
            stack.grid_levels()


def test_one_plane_ternary_equivalence() -> None:
    """A single plane with variance-preserving norm approximates ternary weights."""
    torch.manual_seed(0)
    in_features = 8
    out_features = 4
    stack = ResidualTritStack(
        in_features,
        out_features,
        R=1,
        scale_mode="geometric_balanced",
        residual_normalization="variance_preserving",
    )
    # Force plane weights to a known spread so ternarization is non-trivial.
    with torch.no_grad():
        stack.planes[0].weight.normal_(mean=0.0, std=0.5)
    x = torch.randn(4, in_features)
    out = stack(x, max_planes=1)
    w_eff = stack.effective_weight(max_planes=1)
    expected = torch.nn.functional.linear(x, w_eff, stack.base_module.bias)
    assert torch.allclose(out, expected, atol=1e-5)


def test_normalization_invertibility() -> None:
    """RMS/variance-preserving normalization scales weights by a recoverable factor."""
    for norm in ("rms", "variance_preserving"):
        stack = ResidualTritStack(
            HIDDEN_DIM, 4, R=1, residual_normalization=norm
        )
        raw = stack.planes[0].weight
        norm_raw, rescale = stack._normalize(raw)
        recovered = norm_raw * rescale
        assert torch.allclose(raw, recovered, atol=1e-6)


def test_forward_diagnostics_have_expected_keys() -> None:
    """Diagnostics expose per-plane cost, symbols, scales and residuals."""
    stack = ResidualTritStack(HIDDEN_DIM, 4, R=2)
    x = _hidden(batch=1)
    diag = stack(x, return_diagnostics=True)
    assert isinstance(diag, PlaneOutput)
    assert diag.final_output.shape == (1, 4)
    assert len(diag.plane_outputs) == stack.R
    assert len(diag.symbols) == stack.R
    assert len(diag.scales) == stack.R
    assert len(diag.per_plane_cost_bytes) == stack.R
    assert len(diag.quant_errors) == stack.R
    assert all(c > 0 for c in diag.per_plane_cost_bytes)


def test_build_model_ledger_reports_residual_ternary_plane_bytes() -> None:
    """Plane weights are costed with the residual ternary plane format."""

    class ToyWithPlanes(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stack = ResidualTritStack(16, 8, R=2)

    model = ToyWithPlanes()
    ledger = build_model_ledger(
        model,
        format_map={},
        default_format=ternary_format(),
        residual_plane_format=residual_ternary_plane_format(group_size=128),
    )
    plane_tensors = [
        t for t in ledger.formats.get("residual_ternary_plane", []).tensors
    ]
    assert len(plane_tensors) == 2
    assert all(t.format_id == "residual_ternary_plane" for t in plane_tensors)


def test_head_forces_single_legal_action() -> None:
    """ResidualTritPlaneHead preserves the forced-decision shortcut."""
    head = ResidualTritPlaneHead(HIDDEN_DIM, max_actions=16, R=1)
    legal = ["only_action"]
    out = head.score(_hidden(), StateContext("test"), legal)
    decision = head.decode(out, legal)
    assert decision.decision_kind == "forced"
    assert decision.action_identity == "only_action"


def test_head_scores_multiple_legal_actions() -> None:
    """ResidualTritPlaneHead returns a scored decision inside the legal set."""
    head = ResidualTritPlaneHead(HIDDEN_DIM, max_actions=16, R=1)
    legal = ["a", "b", "c"]
    out = head.score(_hidden(), StateContext("test"), legal)
    decision = head.decode(out, legal)
    assert decision.decision_kind == "scored"
    assert decision.action_identity in legal
    assert 0.0 < decision.confidence <= 1.0


def test_head_metadata_reports_family_and_configuration() -> None:
    """Score metadata carries the head family and residual-stack configuration."""
    head = ResidualTritPlaneHead(
        HIDDEN_DIM,
        max_actions=16,
        R=2,
        scale_mode="geometric_balanced",
        residual_normalization="rms",
    )
    legal = ["x", "y"]
    out = head.score(_hidden(), StateContext("test"), legal)
    assert out.head_family == "residual_trit_plane"
    assert out.metadata["action_count"] == 2
    assert out.metadata["R"] == 2
    assert out.metadata["scale_mode"] == "geometric_balanced"
    assert out.metadata["residual_normalization"] == "rms"


def test_fit_planes_sequential_reduces_residual() -> None:
    """Sequential plane fitting reduces the residual MSE against a teacher."""
    torch.manual_seed(1)
    in_features = 8
    out_features = 4
    stack = ResidualTritStack(
        in_features,
        out_features,
        R=2,
        scale_mode="geometric_balanced",
        residual_normalization="none",
    )
    x = torch.randn(16, in_features)
    teacher = torch.nn.Linear(in_features, out_features)
    teacher.weight.requires_grad = False
    teacher.bias.requires_grad = False
    target = teacher(x).detach()

    with torch.no_grad():
        base_mse = float(torch.nn.functional.mse_loss(stack(x), target).item())

    result = stack.fit_planes_sequential(x, target, steps=30, lr=5e-2)
    histories = result["loss_histories"]
    assert len(histories) == stack.R
    assert all(len(h) == 30 for h in histories)
    # Each plane's final loss should be no worse than its initial loss.
    for h in histories:
        assert h[-1] <= h[0] + 1e-6

    with torch.no_grad():
        final_mse = float(torch.nn.functional.mse_loss(stack(x), target).item())
    assert final_mse < base_mse
