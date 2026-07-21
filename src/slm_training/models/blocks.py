"""Transformer building blocks for TwoTower."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt_()
        return self.weight * x * norm


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, D] queries
        ctx: optional [B, S, D] for cross-attn (defaults to self-attn)
        key_padding_mask: [B, S] True where PAD (ignored)
        return_weights: also return head-averaged attention [B, T, S]
        (explicit softmax path; the SDPA fast path is used otherwise).
        """
        context = ctx if ctx is not None else x
        bsz, tlen, _ = x.shape
        slen = context.size(1)

        q = self.q_proj(x).view(bsz, tlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(context)
            .view(bsz, slen, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(context)
            .view(bsz, slen, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, S] True where PAD → additive -inf mask [B,1,1,S]
            pad = key_padding_mask
            attn_mask = torch.zeros(bsz, 1, 1, slen, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(pad[:, None, None, :], float("-inf"))

        if return_weights:
            # E71: explicit attention for dependency-cluster construction.
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores + attn_mask
            weights = F.softmax(scores, dim=-1)
            if self.dropout and self.training:
                weights = F.dropout(weights, p=self.dropout)
            out = weights @ v
            out = out.transpose(1, 2).contiguous().view(bsz, tlen, -1)
            return self.out_proj(out), weights.mean(dim=1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, tlen, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        self.cross_attn = (
            MultiheadAttention(d_model, n_heads, dropout) if cross_attn else None
        )
        self.norm_cross = RMSNorm(d_model) if cross_attn else None
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        self_pad_mask: torch.Tensor | None = None,
        ctx: torch.Tensor | None = None,
        ctx_pad_mask: torch.Tensor | None = None,
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
        x = x + self.dropout(self.mlp(self.norm2(x)))
        if return_self_attn:
            assert attn_weights is not None
            return x, attn_weights
        return x


class TokenEncoder(nn.Module):
    """Frozen-capable context tower: bidirectional encoder over prompt tokens."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, dropout=dropout, cross_attn=False)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        bsz, seq = input_ids.shape
        if seq > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            seq = self.max_len
        pos = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.tok(input_ids) + self.pos(pos)
        pad_mask = input_ids.eq(pad_id)
        for layer in self.layers:
            x = layer(x, self_pad_mask=pad_mask)
        return self.norm(x)


class DenoiserTower(nn.Module):
    """Trainable masked-diffusion / MaskGIT-style denoiser with cross-attn to context."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 512,
        dropout: float = 0.0,
        *,
        kind_ids: list[int] | None = None,
        n_kinds: int = 0,
        tie_output_embedding: bool = True,
    ) -> None:
        super().__init__()
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
            # Non-persistent stub: unused unless factorized embeddings are on.
            # Keeps older (non-V5) checkpoints loadable without kind_lookup.
            self.register_buffer(
                "kind_lookup",
                torch.zeros(max(vocab_size, 1), dtype=torch.long),
                persistent=False,
            )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, dropout=dropout, cross_attn=True)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self._runtime_symbol_features: torch.Tensor | None = None
        self.tie_output_embedding = bool(tie_output_embedding)
        if self.tie_output_embedding:
            # Tie embeddings: shared storage and gradients.
            self.lm_head.weight = self.tok.weight
        else:
            # Untied output head: keep the same initial function by copying the
            # token embedding matrix, but allow independent updates afterwards.
            self.lm_head.weight.data.copy_(self.tok.weight.data)

    def set_runtime_symbol_features(self, features: torch.Tensor | None) -> None:
        """Attach request-local vocabulary-row deltas (not checkpoint state)."""
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
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a noisy canvas without paying for the vocabulary projection."""
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
            # Clamp ids into lookup range for safety with pad overflows.
            safe = noisy_ids.clamp(min=0, max=self.kind_lookup.numel() - 1)
            x = x + self.kind(self.kind_lookup[safe])
        self_pad = noisy_ids.eq(pad_id)
        attn: torch.Tensor | None = None
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            if return_attn and i == last:
                x, attn = layer(
                    x,
                    self_pad_mask=self_pad,
                    ctx=context,
                    ctx_pad_mask=ctx_pad_mask,
                    return_self_attn=True,
                )
            else:
                x = layer(
                    x, self_pad_mask=self_pad, ctx=context, ctx_pad_mask=ctx_pad_mask
                )
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
        """Project hidden states to the full vocabulary or gathered candidates."""
        if self._runtime_symbol_features is not None and hidden.dim() != 3:
            # Compiler/tree scorers project [D] or [N,D] slices. Feature rows
            # are per-request, so this is only well-defined when exactly one
            # request's features are active (the per-row decode paths).
            if int(self._runtime_symbol_features.size(0)) != 1:
                raise ValueError(
                    "runtime symbol features require [B,T,D] hidden states "
                    "when more than one request is active"
                )
            flat = hidden.reshape(1, -1, hidden.size(-1))
            out = self.project(flat, candidate_ids)
            return out.reshape(*hidden.shape[:-1], out.size(-1))
        features = self._features_for_batch(hidden.size(0))
        if candidate_ids is None:
            logits = self.lm_head(hidden)
            if features is not None:
                logits = logits + torch.einsum("btd,bvd->btv", hidden, features)
            return logits
        raw_weight = self.lm_head.weight
        weight = raw_weight() if callable(raw_weight) else raw_weight
        if weight.is_quantized:
            weight = weight.dequantize()
        weight = weight.index_select(0, candidate_ids)
        logits = F.linear(hidden, weight)
        if features is not None:
            selected = features.index_select(1, candidate_ids)
            logits = logits + torch.einsum("btd,bkd->btk", hidden, selected)
        return logits

    def forward(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
        return_attn: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Run the legacy full-vocabulary path (checkpoint-compatible)."""
        encoded = self.encode(
            noisy_ids,
            context,
            pad_id,
            ctx_pad_mask,
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
