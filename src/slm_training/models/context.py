"""Context tower backends: from-scratch TokenEncoder or frozen HF encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from slm_training.models.blocks import TokenEncoder


class ScratchContextEncoder(nn.Module):
    """Bidirectional token encoder over the OpenUI/prompt tokenizer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 256,
        dropout: float = 0.0,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = TokenEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
            dropout=dropout,
        )
        self.pad_id: int | None = None
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward_prompts(
        self,
        prompts: list[str],
        *,
        encode_fn,
        max_len: int,
        pad_id: int,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [encode_fn(p)[:max_len] for p in prompts]
        max_t = max((len(x) for x in ids), default=1)
        batch = torch.full((len(ids), max_t), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(ids):
            if seq:
                batch[i, : len(seq)] = torch.tensor(
                    seq, dtype=torch.long, device=device
                )
        pad_mask = batch.eq(pad_id)
        ctx = self.encoder(batch, pad_id=pad_id)
        return ctx, pad_mask


class HFContextEncoder(nn.Module):
    """Frozen (by default) Hugging Face encoder/decoder-as-encoder for prompts."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        d_model: int = 128,
        max_len: int = 128,
        freeze: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HF context tower requires `transformers`. "
                "Install with: pip install -e '.[hf]'"
            ) from exc

        self.model_name = model_name
        self.revision = revision
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only
        )
        if self.tokenizer.pad_token is None:
            # GPT-style models often lack pad; reuse eos.
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )
        self.backbone = AutoModel.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only
        )
        hidden = int(getattr(self.backbone.config, "hidden_size", d_model))
        self.proj = nn.Linear(hidden, d_model)
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # Frozen-backbone cache: key -> (hidden_cpu [S,H], pad_mask_cpu [S])
        self._backbone_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.cache_backbone: bool = True

    def clear_backbone_cache(self) -> None:
        self._backbone_cache.clear()

    def forward_prompts(
        self,
        prompts: list[str],
        *,
        encode_fn=None,
        max_len: int | None = None,
        pad_id: int | None = None,
        device: str | torch.device = "cpu",
        cache_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = encode_fn, pad_id
        length = max_len or self.max_len
        use_cache = (
            bool(self.cache_backbone)
            and self.freeze
            and cache_keys is not None
            and len(cache_keys) == len(prompts)
        )
        if use_cache:
            assert cache_keys is not None
            missing_idx: list[int] = []
            missing_prompts: list[str] = []
            missing_keys: list[str] = []
            placeholders: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(
                prompts
            )
            for i, (key, prompt) in enumerate(zip(cache_keys, prompts)):
                hit = self._backbone_cache.get(key)
                if hit is not None:
                    placeholders[i] = hit
                else:
                    missing_idx.append(i)
                    missing_prompts.append(prompt)
                    missing_keys.append(key)
            if missing_prompts:
                encoded = self.tokenizer(
                    missing_prompts,
                    padding=True,
                    truncation=True,
                    max_length=length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.no_grad():
                    out = self.backbone(**encoded)
                    hidden_b = out.last_hidden_state
                pad_b = encoded["attention_mask"].eq(0)
                for j, key in enumerate(missing_keys):
                    row_h = hidden_b[j].detach().cpu()
                    row_p = pad_b[j].detach().cpu()
                    # Trim trailing pad for compact cache.
                    valid = int((~row_p).sum().item()) or 1
                    row_h = row_h[:valid].contiguous()
                    row_p = row_p[:valid].contiguous()
                    self._backbone_cache[key] = (row_h, row_p)
                    placeholders[missing_idx[j]] = (row_h, row_p)
            # Pad batch of variable-length cached rows.
            assert all(p is not None for p in placeholders)
            max_s = max(int(p[0].size(0)) for p in placeholders)  # type: ignore[index]
            hidden_dim = int(placeholders[0][0].size(-1))  # type: ignore[index]
            hidden = torch.zeros(
                len(prompts),
                max_s,
                hidden_dim,
                device=device,
                dtype=self.proj.weight.dtype,
            )
            pad_mask = torch.ones(len(prompts), max_s, dtype=torch.bool, device=device)
            for i, packed in enumerate(placeholders):
                assert packed is not None
                row_h, row_p = packed
                sl = row_h.size(0)
                hidden[i, :sl] = row_h.to(device=device, dtype=hidden.dtype)
                pad_mask[i, :sl] = row_p.to(device=device)
            ctx = self.proj(hidden)
            return ctx, pad_mask

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.set_grad_enabled((not self.freeze) and self.training):
            out = self.backbone(**encoded)
            hidden = out.last_hidden_state
        ctx = self.proj(hidden)
        pad_mask = encoded["attention_mask"].eq(0)
        return ctx, pad_mask

    def trainable_parameter_names(self) -> list[str]:
        names = ["proj.weight", "proj.bias"]
        if not self.freeze:
            names.extend(n for n, _ in self.backbone.named_parameters())
        return names


def build_context_encoder(
    *,
    backend: str,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    max_len: int,
    dropout: float,
    freeze: bool,
    hf_model_name: str | None,
    hf_model_revision: str | None = None,
    local_files_only: bool = False,
) -> nn.Module:
    backend = (backend or "scratch").lower()
    if backend in {"scratch", "token", "local"}:
        return ScratchContextEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
            dropout=dropout,
            freeze=freeze,
        )
    if backend in {"hf", "huggingface", "transformers"}:
        if not hf_model_name:
            raise ValueError("hf_model_name is required when context_backend='hf'")
        return HFContextEncoder(
            model_name=hf_model_name,
            revision=hf_model_revision,
            d_model=d_model,
            max_len=max_len,
            freeze=freeze,
            local_files_only=local_files_only,
        )
    raise ValueError(f"unknown context_backend {backend!r}")


def is_hf_context(module: Any) -> bool:
    return isinstance(module, HFContextEncoder)
