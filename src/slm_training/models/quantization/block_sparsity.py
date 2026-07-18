"""CAP4-04: compiler-routed block sparsity and state-family micro-experts.

Deterministic routing keyed by exact compiler state.  A route may select a
block mask or a low-rank residual expert, but it never alters the legal-action
set.  Unknown/missing state families fall back to a shared dense path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.models.quantization.cost import compute_tensor_cost
from slm_training.models.quantization.formats import QuantFormat, block_sparse_ternary_format

if TYPE_CHECKING:
    from slm_training.models.local_action_head import StateContext


@dataclass(frozen=True)
class StateFamilyKey:
    """Stable, versioned key derived from compiler-visible state."""

    version: str
    family_id: str
    signature: tuple[Any, ...] = ()
    coverage: str = "unknown"


class StateFamilyRouter:
    """Map ``StateContext`` to a deterministic integer route.

    The router is stateless except for the runtime mapping table, which assigns
    stable integer indices to newly seen family ids.  Unknown families always
    map to route 0 (the shared dense fallback).
    """

    def __init__(
        self,
        *,
        key_version: str = "cap4-04.v1",
        unknown_family_id: str = "unknown",
    ) -> None:
        self.key_version = key_version
        self.unknown_family_id = unknown_family_id
        self._route_table: dict[str, int] = {unknown_family_id: 0}
        self._next_route = 1

    def key_from_context(
        self,
        ctx: "StateContext",
        coverage: str = "unknown",
    ) -> StateFamilyKey:
        """Build a stable key from a compiler ``StateContext``."""
        family_id = getattr(ctx, "state_family_id", "") or self.unknown_family_id
        if not family_id:
            family_id = self.unknown_family_id
        signature = getattr(ctx, "state_signature", ()) or ()
        return StateFamilyKey(
            version=self.key_version,
            family_id=family_id,
            signature=tuple(signature),
            coverage=coverage,
        )

    def route_index(self, key: StateFamilyKey) -> int:
        """Return a stable integer route for ``key``."""
        fid = key.family_id if key.family_id else self.unknown_family_id
        if fid not in self._route_table:
            self._route_table[fid] = self._next_route
            self._next_route += 1
        return self._route_table[fid]

    def route_for_context(
        self,
        ctx: "StateContext",
        coverage: str = "unknown",
    ) -> int:
        """Convenience: key + route in one call."""
        return self.route_index(self.key_from_context(ctx, coverage))

    @property
    def n_routes(self) -> int:
        return self._next_route

    def reset(self) -> None:
        """Drop learned family assignments; unknown remains route 0."""
        self._route_table = {self.unknown_family_id: 0}
        self._next_route = 1


def _gather_route_outputs(
    x: torch.Tensor,
    route_indices: torch.Tensor,
    route_fn: dict[int, nn.Module],
) -> torch.Tensor:
    """Apply route-specific modules to per-row inputs and reassemble in order.

    ``x`` may be any shape whose first dimension is the batch.  ``route_fn``
    maps a route integer to a callable accepting and returning the same
    non-batch shape as the input slice.
    """
    if x.shape[0] != route_indices.shape[0]:
        raise ValueError("batch dimension must match route_indices length")
    outputs = [None] * x.shape[0]
    unique = torch.unique(route_indices, sorted=True).tolist()
    for r in unique:
        fn = route_fn.get(r)
        if fn is None:
            continue
        mask = route_indices == r
        subset = x[mask]
        out_subset = fn(subset)
        for idx, place in enumerate(mask.nonzero(as_tuple=False).flatten().tolist()):
            outputs[place] = out_subset[idx]
    if any(o is None for o in outputs):
        raise RuntimeError("some routes had no module in route_fn")
    return torch.stack(outputs, dim=0)


class BlockMaskedLinear(nn.Module):
    """Linear layer whose weight blocks are masked per state-family route.

    Block shape is ``(block_size, block_size)``.  The whole weight matrix is
    always stored; the mask only zeroes blocks at runtime.  This is a reference
    implementation: no optimized sparse kernel is claimed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_routes: int,
        block_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features % block_size != 0 or out_features % block_size != 0:
            raise ValueError(
                f"in_features ({in_features}) and out_features ({out_features}) "
                f"must be divisible by block_size ({block_size})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.n_routes = n_routes
        self.block_size = block_size
        self.out_blocks = out_features // block_size
        self.in_blocks = in_features // block_size

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        # Binary block masks, one per route.  Initialized to all-active.
        self.register_buffer(
            "block_masks",
            torch.ones(n_routes, self.out_blocks, self.in_blocks),
        )

    def _expand_mask(self, route_indices: torch.Tensor) -> torch.Tensor:
        """Return a [B, out_features, in_features] mask for the given routes."""
        selected = self.block_masks[route_indices]  # [B, out_blocks, in_blocks]
        # Upsample to element-wise mask.
        return (
            selected.unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, 1, self.block_size, self.block_size)
            .reshape(-1, self.out_features, self.in_features)
        )

    def forward(self, x: torch.Tensor, route_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Forward with optional per-item route selection."""
        if route_indices is None:
            return F.linear(x, self.weight, self.bias)
        if x.shape[0] != route_indices.shape[0]:
            raise ValueError("batch dimension must match route_indices length")

        mask = self._expand_mask(route_indices)
        masked_weight = self.weight * mask

        input_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1, self.in_features)
        out = torch.bmm(x_flat, masked_weight.transpose(1, 2))
        out = out.reshape(*input_shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def active_blocks(self, route_indices: torch.Tensor) -> torch.Tensor:
        """Number of active blocks for each route in the batch."""
        selected = self.block_masks[route_indices]
        return selected.sum(dim=(-2, -1))

    def total_blocks(self) -> int:
        return self.out_blocks * self.in_blocks

    def set_route_mask(self, route: int, mask: torch.Tensor) -> None:
        """Set the block mask for ``route``; ``mask`` shape [out_blocks, in_blocks]."""
        if route < 0 or route >= self.n_routes:
            raise IndexError(f"route {route} out of range [0, {self.n_routes})")
        if mask.shape != (self.out_blocks, self.in_blocks):
            raise ValueError(
                f"mask shape {mask.shape} != {(self.out_blocks, self.in_blocks)}"
            )
        self.block_masks[route] = mask.to(dtype=self.block_masks.dtype, device=self.block_masks.device)


class StateFamilyExpert(nn.Module):
    """Low-rank residual expert bank: ``y = W_shared x + b + U_r(V_r x)``.

    One down/up pair per route.  Only the active routes in a batch are
    computed.  Unknown route 0 uses the shared path only (no expert residual).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_routes: int,
        rank: int,
        bias: bool = True,
        add_to_shared: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_routes = n_routes
        self.rank = rank
        self.add_to_shared = add_to_shared

        self.shared_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.shared_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("shared_bias", None)

        self.down = nn.ModuleList(
            [nn.Linear(in_features, rank, bias=False) for _ in range(n_routes)]
        )
        self.up = nn.ModuleList(
            [nn.Linear(rank, out_features, bias=False) for _ in range(n_routes)]
        )
        # Zero-initialize expert residuals so the network starts at the shared path.
        for up in self.up:
            nn.init.zeros_(up.weight)

    def forward(self, x: torch.Tensor, route_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Forward with optional per-item route selection."""
        shared = F.linear(x, self.shared_weight, self.shared_bias)
        if route_indices is None or not self.add_to_shared:
            return shared
        if x.shape[0] != route_indices.shape[0]:
            raise ValueError("batch dimension must match route_indices length")

        out = shared.clone()
        input_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1, self.in_features)
        shared_flat = out.reshape(x.shape[0], -1, self.out_features)

        unique = torch.unique(route_indices, sorted=True).tolist()
        for r in unique:
            if r == 0:
                # Route 0 = unknown / shared-only fallback.
                continue
            mask = route_indices == r
            subset = x_flat[mask]
            residual = self.up[r](self.down[r](subset))
            shared_flat[mask] = shared_flat[mask] + residual

        return shared_flat.reshape(*input_shape[:-1], self.out_features)

    def active_params(self, route_indices: torch.Tensor) -> int:
        """Total parameters touched by the routes in the batch."""
        shared = self.shared_weight.numel()
        if self.shared_bias is not None:
            shared += self.shared_bias.numel()
        unique = torch.unique(route_indices).tolist()
        expert = 0
        for r in unique:
            if r == 0:
                continue
            expert += self.down[r].weight.numel() + self.up[r].weight.numel()
        return shared + expert

    def total_params(self) -> int:
        """Total parameters across all routes."""
        shared = self.shared_weight.numel()
        if self.shared_bias is not None:
            shared += self.shared_bias.numel()
        expert = sum(
            p.numel() for m in [self.down, self.up] for p in m.parameters()
        )
        return shared + expert


class CompilerRoutedMLP(nn.Module):
    """MLP with either block-masked or low-rank-expert linear layers."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        n_routes: int,
        block_size: int,
        *,
        expert_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.use_experts = expert_rank is not None
        if self.use_experts:
            self.fc = StateFamilyExpert(
                d_model, hidden_dim, n_routes, rank=expert_rank, bias=True
            )
            self.proj = StateFamilyExpert(
                hidden_dim, d_model, n_routes, rank=expert_rank, bias=True
            )
        else:
            self.fc = BlockMaskedLinear(
                d_model, hidden_dim, n_routes, block_size=block_size, bias=True
            )
            self.proj = BlockMaskedLinear(
                hidden_dim, d_model, n_routes, block_size=block_size, bias=True
            )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, route_indices: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc(x, route_indices)
        h = self.activation(h)
        return self.proj(h, route_indices)


class CompilerRoutedTransformerBlock(nn.Module):
    """Transformer block with a compiler-routed MLP."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_routes: int,
        block_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cross_attn: bool = False,
        *,
        expert_rank: int | None = None,
    ) -> None:
        super().__init__()
        from slm_training.models.blocks import MultiheadAttention, RMSNorm

        self.self_attn = MultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        self.cross_attn = (
            MultiheadAttention(d_model, n_heads, dropout) if cross_attn else None
        )
        self.norm_cross = RMSNorm(d_model) if cross_attn else None
        hidden = int(d_model * mlp_ratio)
        self.mlp = CompilerRoutedMLP(
            d_model,
            hidden,
            n_routes,
            block_size,
            expert_rank=expert_rank,
        )
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        self_pad_mask: torch.Tensor | None = None,
        ctx: torch.Tensor | None = None,
        ctx_pad_mask: torch.Tensor | None = None,
        route_indices: torch.Tensor | None = None,
        *,
        return_self_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attn_weights: torch.Tensor | None = None
        if return_self_attn:
            attn_out, attn_weights = self.self_attn(
                self.norm1(x), key_padding_mask=self_pad_mask, return_weights=True
            )
        else:
            attn_out = self.self_attn(self.norm1(x), key_padding_mask=self_pad_mask)
        x = x + self.dropout(attn_out)
        if self.cross_attn is not None and ctx is not None:
            assert self.norm_cross is not None
            x = x + self.dropout(
                self.cross_attn(
                    self.norm_cross(x), ctx=ctx, key_padding_mask=ctx_pad_mask
                )
            )
        x = x + self.dropout(self.mlp(self.norm2(x), route_indices))
        if return_self_attn:
            assert attn_weights is not None
            return x, attn_weights
        return x


class CompilerRoutedDenoiserTower(nn.Module):
    """Denoiser tower whose MLP layers are compiler-routed.

    Mirrors ``DenoiserTower`` but accepts per-item ``route_indices`` in
    ``encode``/``forward``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 512,
        dropout: float = 0.0,
        *,
        n_routes: int = 4,
        block_size: int = 32,
        expert_rank: int | None = None,
        kind_ids: list[int] | None = None,
        n_kinds: int = 0,
    ) -> None:
        super().__init__()
        from slm_training.models.blocks import RMSNorm

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.kind: nn.Embedding | None = None
        if kind_ids is not None and n_kinds > 0:
            self.kind = nn.Embedding(n_kinds, d_model)
            lookup = torch.tensor(
                [int(k) for k in kind_ids[:vocab_size]]
                + [0] * max(0, vocab_size - len(kind_ids)),
                dtype=torch.long,
            )
            self.register_buffer("kind_lookup", lookup, persistent=True)
        else:
            self.register_buffer(
                "kind_lookup",
                torch.zeros(max(vocab_size, 1), dtype=torch.long),
                persistent=False,
            )
        self.layers = nn.ModuleList(
            [
                CompilerRoutedTransformerBlock(
                    d_model,
                    n_heads,
                    n_routes,
                    block_size,
                    dropout=dropout,
                    cross_attn=True,
                    expert_rank=expert_rank,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.n_routes = n_routes
        self._runtime_symbol_features: torch.Tensor | None = None
        self.lm_head.weight = self.tok.weight

    def set_runtime_symbol_features(self, features: torch.Tensor | None) -> None:
        self._runtime_symbol_features = features

    def _features_for_batch(self, batch_size: int) -> torch.Tensor | None:
        features = self._runtime_symbol_features
        if features is None:
            return None
        if features.size(0) == batch_size:
            return features
        if features.size(0) == 1:
            return features.expand(batch_size, -1, -1)
        raise ValueError(
            f"runtime symbol feature batch {features.size(0)} != {batch_size}"
        )

    def encode(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
        route_indices: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        bsz, seq = noisy_ids.shape
        if seq > self.max_len:
            noisy_ids = noisy_ids[:, : self.max_len]
            seq = self.max_len
        pos = torch.arange(seq, device=noisy_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.tok(noisy_ids) + self.pos(pos)
        features = self._features_for_batch(bsz)
        if features is not None:
            row = torch.arange(bsz, device=noisy_ids.device).unsqueeze(1)
            x = x + features[row, noisy_ids.clamp(0, features.size(1) - 1)]
        if self.kind is not None:
            safe = noisy_ids.clamp(min=0, max=self.kind_lookup.numel() - 1)
            x = x + self.kind(self.kind_lookup[safe])
        self_pad = noisy_ids.eq(pad_id)
        attn: torch.Tensor | None = None
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            kwargs = {}
            if return_attn and i == last:
                kwargs["return_self_attn"] = True
            out = layer(
                x,
                self_pad_mask=self_pad,
                ctx=context,
                ctx_pad_mask=ctx_pad_mask,
                route_indices=route_indices,
                **kwargs,
            )
            if return_attn and i == last:
                x, attn = out  # type: ignore[assignment]
            else:
                x = out  # type: ignore[assignment]
        hidden = self.norm(x)
        if return_attn:
            assert attn is not None
            return hidden, attn
        return hidden

    def project(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_ids is None:
            return self.lm_head(hidden)
        weight = self.lm_head.weight.index_select(0, candidate_ids)
        return F.linear(hidden, weight)

    def forward(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
        route_indices: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
        return_attn: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        encoded = self.encode(
            noisy_ids,
            context,
            pad_id,
            ctx_pad_mask,
            route_indices,
            return_attn=return_attn,
        )
        if return_attn:
            hidden, attn = encoded
        else:
            hidden = encoded
            attn = None
        logits = self.project(hidden)
        if return_attn:
            assert attn is not None
            return logits, hidden, attn
        if return_hidden:
            return logits, hidden
        return logits


def _active_mask_for_param(
    module: nn.Module,
    name: str,
    param: torch.nn.Parameter,
    route_indices: torch.Tensor,
) -> torch.Tensor | None:
    """Build a boolean active mask over one parameter for the given routes."""
    if isinstance(module, BlockMaskedLinear):
        if name == "weight":
            # Union of active blocks across the routes in the batch.
            union = (module.block_masks[route_indices].sum(dim=0) > 0).float()
            mask = (
                union.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, module.block_size, module.block_size)
                .reshape(module.out_features, module.in_features)
            )
            return mask > 0
        # Bias is always active.
        return None
    if isinstance(module, StateFamilyExpert):
        if name == "shared_weight" or name == "shared_bias":
            return None
        # down.<r>.weight and up.<r>.weight are active only when route r appears.
        for r in range(module.n_routes):
            if name in (f"down.{r}.weight", f"up.{r}.weight"):
                active = (route_indices == r).any().item()
                return torch.full_like(param, active, dtype=torch.bool)
        return None
    return None


def compute_block_sparse_cost(
    module: nn.Module,
    route_indices: torch.Tensor,
    fmt: QuantFormat | None = None,
    group_size: int = 128,
    metadata_overhead_bytes: int = 32,
) -> dict[str, Any]:
    """Return active and total byte costs for a block-sparse/expert module.

    This is a helper that callers can use when ``build_model_ledger`` does not
    yet know how to introspect routed modules.
    """
    if fmt is None:
        fmt = block_sparse_ternary_format(group_size=group_size)
    costs: list[Any] = []
    for name, param in module.named_parameters():
        active_mask = _active_mask_for_param(module, name, param, route_indices)
        cost = compute_tensor_cost(
            name=name,
            tensor=param,
            fmt=fmt,
            group_size=group_size,
            metadata_overhead_bytes=metadata_overhead_bytes,
            active_mask=active_mask,
        )
        costs.append(cost)

    total_numel = sum(c.numel for c in costs)
    active_numel = sum(c.active_numel if c.active_numel is not None else c.numel for c in costs)
    total_bytes = sum(c.total_bytes for c in costs)
    active_bytes = sum(
        c.active_total_bytes if c.active_total_bytes is not None else c.total_bytes
        for c in costs
    )
    return {
        "format_id": fmt.format_id,
        "total_numel": total_numel,
        "active_numel": active_numel,
        "active_ratio": active_numel / max(1, total_numel),
        "total_bytes": total_bytes,
        "active_bytes": active_bytes,
    }
