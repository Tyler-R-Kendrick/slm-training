"""DiffuLLaMA-style AR→masked-denoiser adaptation (Track B4 baseline).

Reuses a pretrained HF causal-LM backbone (default SmolLM2-135M, the same
model already serving as the frozen context tower) as the *denoiser* tower:

- the causal mask is replaced with full bidirectional visibility via an
  explicit 4D attention mask (the core DiffuGPT/DiffuLLaMA move,
  arXiv:2410.17891 — their attention-mask annealing and training recipe are
  NOT reproduced);
- fresh OpenUI-vocabulary embeddings and a weight-tied ``lm_head`` replace the
  backbone's original vocabulary;
- the frozen context tower's hiddens are linearly projected and prepended as
  prefix states (the backbone has no cross-attention, unlike
  ``DenoiserTower``).

The class satisfies the exact ``DenoiserTower`` interface consumed by
``TwoTowerModel`` (``forward``/``encode``/``project``/
``set_runtime_symbol_features`` plus ``.tok``/``.kind``/``.lm_head``/
``.max_len``/``.layers``), so training, masking, and every decode path work
unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HFDenoiserTower(nn.Module):
    """Bidirectional masked denoiser on a pretrained HF causal-LM backbone."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        *,
        hf_model_name: str,
        hf_model_revision: str | None = None,
        local_files_only: bool = False,
        kind_ids: list[int] | None = None,
        n_kinds: int = 0,
        tie_output_embedding: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - env guard
            raise RuntimeError(
                "denoiser_backend='hf' requires the [hf] extra (transformers)"
            ) from exc
        self.backbone = AutoModel.from_pretrained(
            hf_model_name,
            revision=hf_model_revision,
            local_files_only=local_files_only,
        )
        hidden = int(self.backbone.config.hidden_size)
        self.hidden_size = hidden
        self.tok = nn.Embedding(vocab_size, hidden)
        self.kind: nn.Embedding | None = None
        if kind_ids is not None and n_kinds > 0:
            self.kind = nn.Embedding(n_kinds, hidden)
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
        self.ctx_proj = nn.Linear(d_model, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.max_len = max_len
        self._runtime_symbol_features: torch.Tensor | None = None
        self.tie_output_embedding = bool(tie_output_embedding)
        if self.tie_output_embedding:
            self.lm_head.weight = self.tok.weight
        else:
            self.lm_head.weight.data.copy_(self.tok.weight.data)

    @property
    def layers(self) -> nn.ModuleList:
        found = getattr(self.backbone, "layers", None)
        if isinstance(found, nn.ModuleList):
            return found
        found = getattr(getattr(self.backbone, "model", None), "layers", None)
        if isinstance(found, nn.ModuleList):
            return found
        return nn.ModuleList()

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

    def _bidirectional_mask(
        self,
        ctx_pad_mask: torch.Tensor | None,
        ctx_len: int,
        self_pad: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Full-visibility 4D float mask that only hides padded key positions.

        Passing a 4D mask overrides the backbone's internal causal mask, which
        is what turns the AR transformer into a bidirectional denoiser.
        """
        bsz, tgt_len = self_pad.shape
        if ctx_pad_mask is None:
            ctx_pad = torch.zeros(
                bsz, ctx_len, dtype=torch.bool, device=self_pad.device
            )
        else:
            ctx_pad = ctx_pad_mask.bool()
        key_pad = torch.cat([ctx_pad, self_pad], dim=1)
        total = key_pad.size(1)
        mask = torch.zeros(bsz, 1, total, total, dtype=dtype, device=self_pad.device)
        mask.masked_fill_(key_pad[:, None, None, :], torch.finfo(dtype).min)
        return mask

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
        x = self.tok(noisy_ids)
        features = self._features_for_batch(bsz)
        if features is not None:
            row = torch.arange(bsz, device=noisy_ids.device).unsqueeze(1)
            x = x + features[row, noisy_ids.clamp(0, features.size(1) - 1)]
        if self.kind is not None:
            safe = noisy_ids.clamp(min=0, max=self.kind_lookup.numel() - 1)
            x = x + self.kind(self.kind_lookup[safe])
        prefix = self.ctx_proj(context)
        ctx_len = prefix.size(1)
        embeds = torch.cat([prefix, x], dim=1)
        self_pad = noisy_ids.eq(pad_id)
        mask = self._bidirectional_mask(ctx_pad_mask, ctx_len, self_pad, embeds.dtype)
        out = self.backbone(
            inputs_embeds=embeds,
            attention_mask=mask,
            output_attentions=return_attn,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state[:, ctx_len:, :]
        if return_attn:
            attn = out.attentions[-1][:, :, ctx_len:, ctx_len:]
            return hidden, attn
        return hidden

    def project(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project hidden states to the full vocabulary or gathered candidates."""
        features = self._features_for_batch(hidden.size(0))
        if candidate_ids is None:
            logits = self.lm_head(hidden)
            if features is not None:
                logits = logits + torch.einsum("btd,bvd->btv", hidden, features)
            return logits
        weight = self.lm_head.weight.index_select(0, candidate_ids)
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
