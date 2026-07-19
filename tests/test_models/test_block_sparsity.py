"""Regression tests for CAP4-04 compiler-routed block sparsity and experts."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.local_action_head import StateContext
from slm_training.models.quantization import (
    BlockMaskedLinear,
    CompilerRoutedDenoiserTower,
    CompilerRoutedMLP,
    CompilerRoutedTransformerBlock,
    StateFamilyExpert,
    StateFamilyRouter,
    block_sparse_ternary_format,
    compute_block_sparse_cost,
    state_family_expert_format,
)

HIDDEN_DIM = 32


def _hidden(batch: int = 1, dim: int = HIDDEN_DIM) -> torch.Tensor:
    return torch.randn(batch, dim)


class TestStateFamilyRouter:
    """Deterministic compiler-state routing."""

    def test_unknown_family_maps_to_zero(self) -> None:
        router = StateFamilyRouter()
        ctx = StateContext("")
        assert router.route_for_context(ctx) == 0

    def test_same_family_same_route(self) -> None:
        router = StateFamilyRouter()
        ctx = StateContext("card_root", state_signature=("component", "root"))
        r1 = router.route_for_context(ctx)
        r2 = router.route_for_context(ctx)
        assert r1 == r2
        assert r1 != 0

    def test_different_families_different_routes(self) -> None:
        router = StateFamilyRouter()
        r1 = router.route_for_context(StateContext("a"))
        r2 = router.route_for_context(StateContext("b"))
        assert r1 != r2

    def test_key_is_stable_and_versioned(self) -> None:
        router = StateFamilyRouter(key_version="test.v1")
        ctx = StateContext("f", state_signature=(1, 2))
        key = router.key_from_context(ctx, coverage="complete")
        assert key.version == "test.v1"
        assert key.family_id == "f"
        assert key.signature == (1, 2)
        assert key.coverage == "complete"


class TestBlockMaskedLinear:
    """Block-masked linear routing."""

    def test_block_size_must_divide(self) -> None:
        with pytest.raises(ValueError, match="divisible by block_size"):
            BlockMaskedLinear(30, 30, n_routes=2, block_size=8)

    def test_dense_forward_matches_regular_linear(self) -> None:
        layer = BlockMaskedLinear(32, 32, n_routes=2, block_size=8)
        x = _hidden(batch=4, dim=32)
        dense_out = layer(x, route_indices=None)
        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
        assert torch.allclose(dense_out, expected, atol=1e-6)

    def test_all_active_mask_is_dense_equivalent(self) -> None:
        layer = BlockMaskedLinear(32, 32, n_routes=2, block_size=8)
        layer.block_masks[1] = torch.ones_like(layer.block_masks[1])
        x = _hidden(batch=4, dim=32)
        routes = torch.zeros(4, dtype=torch.long)
        # Route 0 default all-active, route 1 explicit all-active.
        out0 = layer(x, routes)
        out1 = layer(x, torch.ones(4, dtype=torch.long))
        assert torch.allclose(out0, out1, atol=1e-6)

    def test_zero_blocks_zero_output(self) -> None:
        layer = BlockMaskedLinear(32, 32, n_routes=2, block_size=8)
        layer.block_masks[1] = torch.zeros_like(layer.block_masks[1])
        x = _hidden(batch=2, dim=32)
        out = layer(x, torch.ones(2, dtype=torch.long))
        assert torch.allclose(out, layer.bias, atol=1e-6)

    def test_active_blocks_count(self) -> None:
        layer = BlockMaskedLinear(32, 32, n_routes=2, block_size=8)
        mask = torch.zeros(4, 4)
        mask[0, 0] = 1.0
        layer.set_route_mask(1, mask)
        routes = torch.tensor([0, 1])
        active = layer.active_blocks(routes)
        assert active[0].item() == 16  # default all-active
        assert active[1].item() == 1


class TestStateFamilyExpert:
    """Low-rank residual expert bank."""

    def test_route_zero_is_shared_only(self) -> None:
        expert = StateFamilyExpert(32, 32, n_routes=3, rank=4)
        x = _hidden(batch=2, dim=32)
        out = expert(x, torch.zeros(2, dtype=torch.long))
        expected = torch.nn.functional.linear(x, expert.shared_weight, expert.shared_bias)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_active_params_less_than_total(self) -> None:
        expert = StateFamilyExpert(32, 32, n_routes=4, rank=4)
        routes = torch.tensor([1, 1, 2])
        active = expert.active_params(routes)
        total = expert.total_params()
        assert active < total
        # Shared + routes 1 and 2 only.
        shared = expert.shared_weight.numel() + expert.shared_bias.numel()
        route_params = expert.down[1].weight.numel() + expert.up[1].weight.numel()
        route_params += expert.down[2].weight.numel() + expert.up[2].weight.numel()
        assert active == shared + route_params


class TestCostAccounting:
    """Active versus total byte accounting."""

    def test_block_sparse_cost_reports_active_and_total(self) -> None:
        layer = BlockMaskedLinear(32, 32, n_routes=2, block_size=8)
        # Route 1 uses only one block.
        mask = torch.zeros(4, 4)
        mask[0, 0] = 1.0
        layer.set_route_mask(1, mask)
        routes = torch.tensor([1, 1])
        cost = compute_block_sparse_cost(
            layer, routes, fmt=block_sparse_ternary_format()
        )
        assert cost["active_numel"] < cost["total_numel"]
        assert cost["active_bytes"] < cost["total_bytes"]
        assert 0.0 < cost["active_ratio"] < 1.0

    def test_expert_cost_reports_active_and_total(self) -> None:
        expert = StateFamilyExpert(32, 32, n_routes=4, rank=4)
        routes = torch.tensor([1, 1, 2])
        cost = compute_block_sparse_cost(
            expert, routes, fmt=state_family_expert_format()
        )
        assert cost["active_numel"] < cost["total_numel"]
        assert cost["active_bytes"] < cost["total_bytes"]


class TestCompilerRoutedModules:
    """End-to-end routed transformer pieces."""

    def test_routed_mlp_default_is_dense(self) -> None:
        mlp = CompilerRoutedMLP(32, 128, n_routes=2, block_size=8)
        x = _hidden(batch=2, dim=32)
        out_default = mlp(x)
        out_routed = mlp(x, route_indices=None)
        assert torch.allclose(out_default, out_routed, atol=1e-6)

    def test_routed_transformer_block_runs(self) -> None:
        block = CompilerRoutedTransformerBlock(
            d_model=32,
            n_heads=4,
            n_routes=2,
            block_size=8,
            cross_attn=True,
        )
        x = torch.randn(2, 5, 32)
        ctx = torch.randn(2, 3, 32)
        routes = torch.tensor([0, 1])
        out = block(x, ctx=ctx, route_indices=routes)
        assert out.shape == x.shape

    def test_routed_denoiser_runs(self) -> None:
        denoiser = CompilerRoutedDenoiserTower(
            vocab_size=16,
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_len=16,
            n_routes=3,
            block_size=8,
        )
        noisy = torch.randint(0, 16, (2, 4))
        context = torch.randn(2, 3, 32)
        routes = torch.tensor([0, 1])
        logits = denoiser(noisy, context, pad_id=0, route_indices=routes)
        assert logits.shape == (2, 4, 16)

    def test_routed_denoiser_expert_runs(self) -> None:
        denoiser = CompilerRoutedDenoiserTower(
            vocab_size=16,
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_len=16,
            n_routes=3,
            block_size=8,
            expert_rank=4,
        )
        noisy = torch.randint(0, 16, (2, 4))
        context = torch.randn(2, 3, 32)
        routes = torch.tensor([0, 2])
        logits = denoiser(noisy, context, pad_id=0, route_indices=routes)
        assert logits.shape == (2, 4, 16)
