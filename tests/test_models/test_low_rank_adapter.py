"""Torch-gated tests for the removable low-rank adapter wrapper (LDI2-01 / SLM-123)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from slm_training.models.adapters.low_rank import LowRankAdapter  # noqa: E402


def _base() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(6, 4)


def _adapter(rank: int = 2, alpha: float = 4.0, dropout: float = 0.0) -> LowRankAdapter:
    return LowRankAdapter(_base(), rank=rank, alpha=alpha, dropout=dropout)


def test_fresh_enabled_adapter_is_bit_identical_to_parent() -> None:
    adapter = _adapter()
    x = torch.randn(3, 6)
    assert torch.equal(adapter(x), adapter.base(x))  # B is zero-initialized


def test_nonzero_adapter_effect_and_disable_restore() -> None:
    adapter = _adapter()
    x = torch.randn(3, 6)
    baseline = adapter.base(x).clone()
    with torch.no_grad():
        adapter.lora_B.fill_(0.1)  # simulate a trained delta
    assert not torch.allclose(adapter(x), baseline)
    adapter.disable_adapter()
    assert torch.equal(adapter(x), baseline)  # disable restores the exact parent map
    adapter.enable_adapter()
    assert not torch.allclose(adapter(x), baseline)


def test_only_adapter_parameters_receive_gradients() -> None:
    adapter = _adapter()
    with torch.no_grad():
        adapter.lora_B.fill_(0.1)
    adapter(torch.randn(2, 6)).sum().backward()
    assert adapter.base.weight.grad is None
    assert adapter.base.bias.grad is None
    assert adapter.lora_A.grad is not None
    assert adapter.lora_B.grad is not None


def test_adapter_params_follow_base_dtype_and_batched_dims() -> None:
    adapter = _adapter()
    assert adapter.lora_A.dtype == adapter.base.weight.dtype
    assert adapter.lora_B.device == adapter.base.weight.device
    out = adapter(torch.randn(2, 3, 6))  # batched leading dims
    assert out.shape == (2, 3, 4)


def test_merged_linear_matches_enabled_output_without_mutating_base() -> None:
    adapter = _adapter()
    with torch.no_grad():
        adapter.lora_B.fill_(0.1)
    base_weight_before = adapter.base.weight.detach().clone()
    x = torch.randn(5, 6)

    merged = adapter.merged_linear()
    assert isinstance(merged, nn.Linear) and not isinstance(merged, LowRankAdapter)
    assert torch.allclose(merged(x), adapter(x), atol=1e-6)
    # Merge is one-way on a copy: the parent weight and the removable adapter are intact.
    assert torch.equal(adapter.base.weight, base_weight_before)
    assert torch.allclose(adapter(x), adapter.base(x) + adapter.scaling * nn.functional.linear(
        nn.functional.linear(x, adapter.lora_A), adapter.lora_B
    ))


def test_rank_must_be_positive_and_wraps_only_linear() -> None:
    with pytest.raises(ValueError, match="rank must be positive"):
        LowRankAdapter(_base(), rank=0, alpha=1.0, dropout=0.0)
    with pytest.raises(TypeError, match="wraps an nn.Linear"):
        LowRankAdapter(nn.ReLU(), rank=2, alpha=1.0, dropout=0.0)  # type: ignore[arg-type]


def test_alpha_and_dropout_are_validated() -> None:
    # Direct construction bypasses TwoTowerAdapterSpec, so the wrapper mirrors its checks.
    for bad_alpha in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="alpha must be a positive finite"):
            LowRankAdapter(_base(), rank=2, alpha=bad_alpha, dropout=0.0)
    for bad_dropout in (-0.1, 1.0, float("nan")):
        with pytest.raises(ValueError, match=r"dropout must be a finite number in \[0, 1\)"):
            LowRankAdapter(_base(), rank=2, alpha=1.0, dropout=bad_dropout)
