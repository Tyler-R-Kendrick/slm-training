"""Semantic connector variants between a frozen context encoder and a sparse grammar-action scorer.

This module is wiring-only for SLM-166 (SDE1-04).  It is intentionally not wired
into ``_choice_ltr_decode_batch``; callers that set ``semantic_connector="none"``
get an identity path that returns the pooled context unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "ConnectorOutput",
    "LinearConnector",
    "LowRankConnector",
    "CrossAttentionConnector",
    "SemanticConnector",
    "count_connector_parameters",
    "estimate_connector_flops",
]


def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean-pool ``x`` over the sequence dimension using ``mask``.

    Args:
        x: Tensor of shape ``(batch, seq, d_model)``.
        mask: Boolean or float tensor of shape ``(batch, seq)``.  ``None`` means
            all positions are valid.

    Returns:
        Tensor of shape ``(batch, d_model)``.
    """
    if mask is None:
        return x.mean(dim=-2)
    mask = mask.to(dtype=x.dtype, device=x.device)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    masked = x * mask
    denom = mask.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    return (masked.sum(dim=-2, keepdim=True) / denom).squeeze(-2)


@dataclass
class ConnectorOutput:
    """Unified output envelope for all connector variants."""

    context_vectors: torch.Tensor
    mask: torch.Tensor | None
    attention_weights: torch.Tensor | None
    connector_type: str
    connector_version: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict (tensors become shapes, not values)."""
        return {
            "context_vectors_shape": list(self.context_vectors.shape),
            "mask_shape": list(self.mask.shape) if self.mask is not None else None,
            "attention_weights_shape": (
                list(self.attention_weights.shape)
                if self.attention_weights is not None
                else None
            ),
            "connector_type": self.connector_type,
            "connector_version": self.connector_version,
        }


class LinearConnector(nn.Module):
    """Masked mean pool followed by an optional linear projection."""

    def __init__(
        self,
        d_model: int,
        *,
        output_dim: int | None = None,
        use_projection: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.output_dim = int(output_dim) if output_dim is not None else self.d_model
        self.use_projection = bool(use_projection)
        if self.use_projection:
            self.projection = nn.Linear(self.d_model, self.output_dim, bias=True)
        else:
            self.projection = None
        self.connector_type = "linear"
        self.connector_version = "v1"

    def forward(
        self, context_vectors: torch.Tensor, mask: torch.Tensor | None = None
    ) -> ConnectorOutput:
        pooled = _masked_mean_pool(context_vectors, mask)
        if self.projection is not None:
            out = self.projection(pooled)
        else:
            out = pooled
        batch_size = out.shape[0]
        out_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=out.device)
        return ConnectorOutput(
            context_vectors=out,
            mask=out_mask,
            attention_weights=None,
            connector_type=self.connector_type,
            connector_version=self.connector_version,
        )


class LowRankConnector(nn.Module):
    """Bottleneck MLP connector with residual connection and layer norm."""

    def __init__(
        self,
        d_model: int,
        *,
        hidden_dim: int = 256,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.down = nn.Linear(self.d_model, self.hidden_dim, bias=True)
        self.up = nn.Linear(self.hidden_dim, self.d_model, bias=True)
        self.norm = nn.LayerNorm(self.d_model)
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"unknown activation: {activation!r}")
        self.connector_type = "low_rank"
        self.connector_version = "v1"

    def forward(
        self, context_vectors: torch.Tensor, mask: torch.Tensor | None = None
    ) -> ConnectorOutput:
        pooled = _masked_mean_pool(context_vectors, mask)
        hidden = self.activation(self.down(pooled))
        out = self.up(hidden) + pooled
        out = self.norm(out)
        batch_size = out.shape[0]
        out_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=out.device)
        return ConnectorOutput(
            context_vectors=out,
            mask=out_mask,
            attention_weights=None,
            connector_type=self.connector_type,
            connector_version=self.connector_version,
        )


class CrossAttentionConnector(nn.Module):
    """Small learned-query cross-attention connector (Q-Former style).

    Uses at most two attention blocks and 4-8 learned queries by default.
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_queries: int = 4,
        n_heads: int = 2,
        n_blocks: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 1 <= n_blocks <= 2:
            raise ValueError(f"n_blocks must be 1 or 2, got {n_blocks}")
        self.d_model = int(d_model)
        self.n_queries = int(n_queries)
        self.n_heads = int(n_heads)
        self.n_blocks = int(n_blocks)
        self.query_embed = nn.Parameter(
            torch.randn(self.n_queries, self.d_model) * 0.02
        )
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(
                            self.d_model,
                            self.n_heads,
                            dropout=float(dropout),
                            batch_first=True,
                        ),
                        "norm": nn.LayerNorm(self.d_model),
                    }
                )
            )
        self.connector_type = "cross_attention"
        self.connector_version = "v1"

    def forward(
        self, context_vectors: torch.Tensor, mask: torch.Tensor | None = None
    ) -> ConnectorOutput:
        batch_size = context_vectors.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.to(dtype=torch.bool, device=context_vectors.device)
        h = queries
        attn_weights = None
        for block in self.blocks:
            attn_out, attn_weights = block["attn"](
                h,
                context_vectors,
                context_vectors,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            h = block["norm"](h + attn_out)
        out_mask = torch.ones(
            batch_size, self.n_queries, dtype=torch.bool, device=h.device
        )
        return ConnectorOutput(
            context_vectors=h,
            mask=out_mask,
            attention_weights=attn_weights,
            connector_type=self.connector_type,
            connector_version=self.connector_version,
        )


class SemanticConnector(nn.Module):
    """Factory module that selects a connector variant by string type.

    Supported types:

    - ``none``: identity path; returns the masked mean-pooled context unchanged.
    - ``linear``: :class:`LinearConnector`.
    - ``low_rank``: :class:`LowRankConnector`.
    - ``cross_attention``: :class:`CrossAttentionConnector`.
    """

    def __init__(
        self,
        connector_type: str = "none",
        *,
        d_model: int = 128,
        connector_hidden_dim: int = 256,
        connector_rank: int = 32,
        connector_n_queries: int = 4,
        connector_freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.connector_type = str(connector_type).lower()
        self.d_model = int(d_model)
        self.connector_hidden_dim = int(connector_hidden_dim)
        self.connector_rank = int(connector_rank)
        self.connector_n_queries = int(connector_n_queries)
        self.connector_freeze_encoder = bool(connector_freeze_encoder)
        if self.connector_type == "none":
            self.connector: nn.Module | None = None
        elif self.connector_type == "linear":
            self.connector = LinearConnector(self.d_model, use_projection=True)
        elif self.connector_type == "low_rank":
            self.connector = LowRankConnector(
                self.d_model, hidden_dim=self.connector_hidden_dim
            )
        elif self.connector_type == "cross_attention":
            self.connector = CrossAttentionConnector(
                self.d_model, n_queries=self.connector_n_queries
            )
        else:
            raise ValueError(f"unknown semantic_connector type: {connector_type!r}")

    def forward(
        self, context_vectors: torch.Tensor, mask: torch.Tensor | None = None
    ) -> ConnectorOutput:
        if self.connector is None:
            pooled = _masked_mean_pool(context_vectors, mask)
            batch_size = pooled.shape[0]
            out_mask = torch.ones(
                batch_size, 1, dtype=torch.bool, device=pooled.device
            )
            return ConnectorOutput(
                context_vectors=pooled,
                mask=out_mask,
                attention_weights=None,
                connector_type="none",
                connector_version="v1",
            )
        output: ConnectorOutput = self.connector(context_vectors, mask)
        return output


def count_connector_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in a connector module."""
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def estimate_connector_flops(
    module: nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
) -> int:
    """Return a rough MAC/FLOP estimate for one connector forward pass.

    The estimate is intentionally conservative and CPU-safe; it is used only for
    fixture-level capacity comparisons, not production profiling.
    """
    if isinstance(module, SemanticConnector):
        module = module.connector
    if module is None or (isinstance(module, nn.Module) and len(list(module.parameters())) == 0):
        return 0

    if isinstance(module, LinearConnector):
        # Mean pooling over seq_len positions plus optional linear projection.
        flops = batch_size * seq_len * d_model
        if module.use_projection:
            flops += batch_size * d_model * module.output_dim
        return flops

    if isinstance(module, LowRankConnector):
        # Mean pooling + down + up projection.
        flops = batch_size * seq_len * d_model
        flops += batch_size * d_model * module.hidden_dim
        flops += batch_size * module.hidden_dim * d_model
        return flops

    if isinstance(module, CrossAttentionConnector):
        # Q @ K^T + softmax @ V for each query; plus output projection.
        n_queries = module.n_queries
        flops = batch_size * n_queries * d_model * seq_len * 2
        flops += batch_size * n_queries * seq_len * d_model
        return flops

    # Fallback: count parameter count times batch size as a crude proxy.
    params = count_connector_parameters(module)
    return batch_size * params
