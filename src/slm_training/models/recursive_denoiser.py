"""Shared-recursive denoiser tower (SLM-138).

A compact denoiser that recurses a small shared transition instead of stacking
independent TransformerBlocks.  It preserves the ``DenoiserTower`` public
contract (``forward`` / ``encode`` / ``project`` / ``set_runtime_symbol_features``
plus ``tok`` / ``kind`` / ``lm_head`` / ``max_len`` / ``layers``) so it can be
dropped into ``TwoTowerModel`` without changing masking, decode, or checkpoint
shapes.

Architecture (V1 primary):
  y_0 = token + position + kind + request-local symbol features
  z_0 = learned latent + projected pooled context + position
  for r in 1..R:
      z_r = z_{r-1} + F_theta(norm(z_{r-1} + y_{r-1}), context)
      y_r = y_{r-1} + G_theta(norm(y_{r-1} + z_r),     context)
      h_r = norm(y_r)
      logits_r = lm_head(h_r)

F_theta and G_theta are built from the same small set of
``TransformerBlock(cross_attn=True)`` instances; they are reused by object
identity every recursion.  With ``recursive_steps=1`` and
``recursive_transition_layers`` equal to the old stacked layer count the tower
has the same parameter count and layer names as ``DenoiserTower`` (the extra
z-state path is new, so it is not byte-identical, but the public contract is
preserved).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.models.blocks import RMSNorm, TransformerBlock


class SharedRecursiveDenoiserTower(nn.Module):
    """Recursive shared-transition denoiser matching the ``DenoiserTower`` contract."""

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
        recursive_steps: int = 1,
        recursive_transition_layers: int | None = None,
        tie_output_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.recursive_steps = max(1, int(recursive_steps))
        self.recursive_transition_layers = (
            recursive_transition_layers
            if recursive_transition_layers is not None
            else n_layers
        )

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
                TransformerBlock(
                    d_model, n_heads, dropout=dropout, cross_attn=True
                )
                for _ in range(self.recursive_transition_layers)
            ]
        )
        # Split the shared transition into the z-update (F) and y-update (G).
        # For n=1 this puts the single block into G, making R=1 behave like a
        # single cross-attention block applied to y+z.
        f_end = self.recursive_transition_layers // 2
        self._f_layers = self.layers[:f_end]
        self._g_layers = self.layers[f_end:]

        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tie_output_embedding = bool(tie_output_embedding)
        if self.tie_output_embedding:
            self.lm_head.weight = self.tok.weight
        else:
            self.lm_head.weight.data.copy_(self.tok.weight.data)

        # z-state path: learned latent + projected pooled context + position.
        self.z_latent = nn.Parameter(torch.zeros(max_len, d_model))
        self.ctx_proj = nn.Linear(d_model, d_model)

        self._runtime_symbol_features: torch.Tensor | None = None

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

    def _apply_layers(
        self,
        x: torch.Tensor,
        layers: list[TransformerBlock] | nn.ModuleList,
        self_pad_mask: torch.Tensor | None,
        ctx: torch.Tensor,
        ctx_pad_mask: torch.Tensor | None,
        *,
        return_last_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply a sequence of TransformerBlocks, optionally returning last attn."""
        attn: torch.Tensor | None = None
        last = len(layers) - 1
        for i, layer in enumerate(layers):
            want_attn = return_last_attn and i == last
            if want_attn:
                out = layer(
                    x,
                    self_pad_mask=self_pad_mask,
                    ctx=ctx,
                    ctx_pad_mask=ctx_pad_mask,
                    return_self_attn=True,
                )
                assert isinstance(out, tuple)
                x, attn = out
            else:
                out = layer(
                    x,
                    self_pad_mask=self_pad_mask,
                    ctx=ctx,
                    ctx_pad_mask=ctx_pad_mask,
                )
                x = out if not isinstance(out, tuple) else out[0]
        if return_last_attn:
            assert attn is not None
            return x, attn
        return x

    def recursive_outputs(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
        return_attn: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Run the full recursive recurrence and expose per-depth outputs.

        Returns a dict with:
          - ``logits``: final logits [B, T, V]
          - ``hidden``: final hidden [B, T, D] (only if ``return_hidden=True``)
          - ``depth_hiddens``: list of [B, T, D] for each recursion step
          - ``depth_logits``: list of [B, T, V] for each recursion step
          - ``attn``: last-layer self-attention [B, T, T] (only if ``return_attn=True``)
        """
        bsz, seq = noisy_ids.shape
        if seq > self.max_len:
            noisy_ids = noisy_ids[:, : self.max_len]
            seq = self.max_len
        pos = torch.arange(seq, device=noisy_ids.device).unsqueeze(0).expand(bsz, -1)

        y = self.tok(noisy_ids) + self.pos(pos)
        features = self._features_for_batch(bsz)
        if features is not None:
            row = torch.arange(bsz, device=noisy_ids.device).unsqueeze(1)
            y = y + features[row, noisy_ids.clamp(0, features.size(1) - 1)]
        if self.kind is not None:
            safe = noisy_ids.clamp(min=0, max=self.kind_lookup.numel() - 1)
            y = y + self.kind(self.kind_lookup[safe])

        self_pad = noisy_ids.eq(pad_id)

        z = self.z_latent[pos]
        if ctx_pad_mask is None:
            pooled = context.mean(dim=1)
        else:
            mask = ctx_pad_mask.logical_not().unsqueeze(-1).to(context.dtype)
            pooled = (context * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        z = z + self.ctx_proj(pooled).unsqueeze(1)
        z = z + self.pos(pos)

        depth_hiddens: list[torch.Tensor] = []
        depth_logits: list[torch.Tensor] = []
        attn: torch.Tensor | None = None

        for r in range(1, self.recursive_steps + 1):
            # z_r = z_{r-1} + F_theta(norm(z_{r-1} + y_{r-1}), context)
            f_in = self.norm(z + y)
            f_out = self._apply_layers(
                f_in, self._f_layers, self_pad, context, ctx_pad_mask
            )
            assert isinstance(f_out, torch.Tensor)
            z = z + f_out

            # y_r = y_{r-1} + G_theta(norm(y_{r-1} + z_r), context)
            g_in = self.norm(y + z)
            return_last_attn = return_attn and r == self.recursive_steps
            g_out = self._apply_layers(
                g_in,
                self._g_layers,
                self_pad,
                context,
                ctx_pad_mask,
                return_last_attn=return_last_attn,
            )
            if return_last_attn and isinstance(g_out, tuple):
                g_out, attn = g_out
            else:
                assert isinstance(g_out, torch.Tensor)
            y = y + g_out

            h = self.norm(y)
            depth_hiddens.append(h)
            depth_logits.append(self.project(h))

        final_hidden = depth_hiddens[-1]
        final_logits = depth_logits[-1]

        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "logits": final_logits,
            "depth_hiddens": depth_hiddens,
            "depth_logits": depth_logits,
        }
        if return_hidden:
            result["hidden"] = final_hidden
        if return_attn and attn is not None:
            result["attn"] = attn
        return result

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
        out = self.recursive_outputs(
            noisy_ids,
            context,
            pad_id,
            ctx_pad_mask,
            return_hidden=True,
            return_attn=return_attn,
        )
        hidden = out["hidden"]
        assert isinstance(hidden, torch.Tensor)
        if return_attn:
            attn = out["attn"]
            assert isinstance(attn, torch.Tensor)
            return hidden, attn
        return hidden

    def project(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project hidden states to the full vocabulary or gathered candidates."""
        if self._runtime_symbol_features is not None and hidden.dim() != 3:
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
        """Run the full-vocabulary path with the same returns as ``DenoiserTower``."""
        out = self.recursive_outputs(
            noisy_ids,
            context,
            pad_id,
            ctx_pad_mask,
            return_hidden=return_hidden or return_attn,
            return_attn=return_attn,
        )
        logits = out["logits"]
        assert isinstance(logits, torch.Tensor)
        if return_attn:
            attn = out["attn"]
            assert isinstance(attn, torch.Tensor)
            hidden = out["hidden"]
            assert isinstance(hidden, torch.Tensor)
            return logits, hidden, attn
        if return_hidden:
            hidden = out["hidden"]
            assert isinstance(hidden, torch.Tensor)
            return logits, hidden
        return logits
