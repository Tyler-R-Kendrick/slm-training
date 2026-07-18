"""Small causal byte-autoregressor for VSS3-05 surface realization.

Generates short surface strings (internal identifiers, decorative text) under
per-slot constraints. Kept separate from the main TwoTower/diffusion paths so
it can be trained independently and fallback to the deterministic baseline is
always available.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PRINTABLE_ASCII = string.printable[:-5]  # exclude \t\n\r\x0b\x0c


@dataclass(frozen=True)
class SurfaceAutoregressorConfig:
    """Hyperparameters for the surface autoregressor."""

    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    max_len: int = 64
    dropout: float = 0.0
    max_bytes: int = 64
    vocab_kind: str = "byte"  # byte only for V1

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_len": self.max_len,
            "dropout": self.dropout,
            "max_bytes": self.max_bytes,
            "vocab_kind": self.vocab_kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurfaceAutoregressorConfig:
        return cls(
            d_model=int(data.get("d_model", 64)),
            n_layers=int(data.get("n_layers", 2)),
            n_heads=int(data.get("n_heads", 2)),
            max_len=int(data.get("max_len", 64)),
            dropout=float(data.get("dropout", 0.0)),
            max_bytes=int(data.get("max_bytes", 64)),
            vocab_kind=str(data.get("vocab_kind", "byte")),
        )


class SurfaceByteVocab:
    """Tiny byte vocabulary for surface strings.

    Tokens are ``<pad>``, ``<bos>``, ``<eos>``, ``<unk>``, and ``B:xx`` for
    printable ASCII codepoints. This makes the model corpus-independent: any
    lowercase identifier or decorative ASCII text can be spelled token-by-token.
    """

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    SPECIAL = (PAD, BOS, EOS, UNK)

    def __init__(self) -> None:
        self.tokens = list(self.SPECIAL)
        self.tokens.extend(f"B:{ord(ch):02x}" for ch in PRINTABLE_ASCII)
        self.token_to_id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id[self.PAD]
        self.bos_id = self.token_to_id[self.BOS]
        self.eos_id = self.token_to_id[self.EOS]
        self.unk_id = self.token_to_id[self.UNK]
        self.vocab_size = len(self.tokens)
        self.byte_to_id = {
            ch: self.token_to_id[f"B:{ord(ch):02x}"] for ch in PRINTABLE_ASCII
        }

    def encode(self, text: str, *, add_special: bool = True) -> list[int]:
        ids = [self.byte_to_id.get(ch, self.unk_id) for ch in text]
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        chars: list[str] = []
        for i in ids:
            tok = self.id_to_token.get(i, self.UNK)
            if skip_special and tok in self.SPECIAL:
                continue
            if tok.startswith("B:"):
                chars.append(chr(int(tok[2:], 16)))
            else:
                chars.append("�")
        return "".join(chars)

    def __len__(self) -> int:
        return self.vocab_size


# Generic identifier grammar: lowercase start, alphanumeric/underscore.
_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


class IdentifierConstraint:
    """Constrained next-token set for OpenUI-style internal identifiers."""

    FIRST = set("abcdefghijklmnopqrstuvwxyz_")
    REST = set("abcdefghijklmnopqrstuvwxyz0123456789_")

    def __init__(
        self,
        vocab: SurfaceByteVocab,
        *,
        max_bytes: int = 64,
        reserved: set[str] | frozenset[str] | None = None,
        peers: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.max_bytes = max_bytes
        self.reserved: frozenset[str] = frozenset(reserved or ())
        self.peers: frozenset[str] = frozenset(peers or ())
        self.first_ids = {vocab.byte_to_id[ch] for ch in self.FIRST if ch in vocab.byte_to_id}
        self.rest_ids = {vocab.byte_to_id[ch] for ch in self.REST if ch in vocab.byte_to_id}

    def allowed_next(self, prefix: str) -> set[int]:
        """Return token ids that may legally follow ``prefix``."""
        if len(prefix.encode("utf-8")) >= self.max_bytes:
            return set()
        if not prefix:
            return self.first_ids.copy()
        allowed = self.rest_ids.copy()
        candidate = prefix
        if (
            _IDENTIFIER_RE.match(candidate)
            and candidate not in self.reserved
            and candidate not in self.peers
            and len(candidate.encode("utf-8")) <= self.max_bytes
        ):
            allowed.add(self.vocab.eos_id)
        return allowed

    def is_complete(self, value: str) -> bool:
        return (
            bool(value)
            and _IDENTIFIER_RE.match(value) is not None
            and value not in self.reserved
            and value not in self.peers
            and len(value.encode("utf-8")) <= self.max_bytes
        )


class DecorativeConstraint:
    """Constrained next-token set for decorative ASCII text."""

    def __init__(self, vocab: SurfaceByteVocab, *, max_bytes: int = 256) -> None:
        self.vocab = vocab
        self.max_bytes = max_bytes
        self.byte_ids = set(vocab.byte_to_id.values())

    def allowed_next(self, prefix: str) -> set[int]:
        if len(prefix.encode("utf-8")) >= self.max_bytes:
            return {self.vocab.eos_id}
        allowed = self.byte_ids.copy()
        allowed.add(self.vocab.eos_id)
        return allowed

    def is_complete(self, value: str) -> bool:
        return len(value.encode("utf-8")) <= self.max_bytes


class CausalSelfAttention(nn.Module):
    """Causal scaled dot-product attention."""

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
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, tlen, _ = x.shape
        q = self.q_proj(x).view(bsz, tlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, tlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, tlen, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal mask: positions j > i are masked.
        causal = torch.triu(torch.ones(tlen, tlen, device=x.device), diagonal=1).bool()
        attn_mask = torch.zeros(bsz, self.n_heads, tlen, tlen, device=x.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(causal[None, None, :, :], float("-inf"))
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :]  # [B,1,1,S]
            attn_mask = attn_mask.masked_fill(pad, float("-inf"))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, tlen, -1)
        return self.out_proj(out)


class SurfaceTransformerBlock(nn.Module):
    """Causal decoder block with optional cross-attention to context."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = (
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            if cross_attn
            else None
        )
        self.norm_cross = nn.LayerNorm(d_model) if cross_attn else None
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        self_pad_mask: torch.Tensor | None = None,
        ctx: torch.Tensor | None = None,
        ctx_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), key_padding_mask=self_pad_mask))
        if self.cross_attn is not None and ctx is not None:
            assert self.norm_cross is not None
            cross_out, _ = self.cross_attn(
                self.norm_cross(x), ctx, ctx, key_padding_mask=ctx_pad_mask
            )
            x = x + self.dropout(cross_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class SurfaceContextEncoder(nn.Module):
    """Tiny encoder that builds a fixed-size context vector from a prompt."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 1,
        n_heads: int = 2,
        max_len: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(
            [SurfaceTransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.max_len = max_len
        self.d_model = d_model

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
        # Mean pool over non-pad positions.
        x = self.norm(x)
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled  # [B, D]


class SurfaceAutoregressor(nn.Module):
    """Causal byte-decoder for surface slot realization."""

    def __init__(self, config: SurfaceAutoregressorConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab = SurfaceByteVocab()
        if config.vocab_kind != "byte":
            raise ValueError(f"unsupported vocab_kind {config.vocab_kind!r}")
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.tok = nn.Embedding(self.vocab.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.max_len, config.d_model)
        self.ctx_encoder = SurfaceContextEncoder(
            vocab_size=self.vocab.vocab_size,
            d_model=config.d_model,
            n_layers=max(1, config.n_layers // 2),
            n_heads=config.n_heads,
            max_len=config.max_len,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleList(
            [
                SurfaceTransformerBlock(
                    config.d_model,
                    config.n_heads,
                    dropout=config.dropout,
                    cross_attn=True,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, self.vocab.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight  # tie weights
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward. Returns logits [B, T, V]."""
        bsz, tlen = input_ids.shape
        ctx = self.ctx_encoder(prompt_ids, pad_id=self.vocab.pad_id)  # [B, D]
        ctx = ctx.unsqueeze(1)  # [B, 1, D] for cross-attn
        pos = torch.arange(tlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.tok(input_ids) + self.pos(pos)
        pad_mask = input_ids.eq(self.vocab.pad_id)
        for layer in self.layers:
            x = layer(x, self_pad_mask=pad_mask, ctx=ctx)
        x = self.norm(x)
        return self.lm_head(x)

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(input_ids, prompt_ids)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.vocab.pad_id,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        constraint: IdentifierConstraint | DecorativeConstraint,
        *,
        max_bytes: int | None = None,
        temperature: float = 0.0,
        top_k: int = 1,
        seed: int | None = None,
    ) -> str | None:
        """Constrained greedy/top-k generation for one slot.

        Returns the generated string, or ``None`` if the decoder reaches a dead
        end before a legal EOS.
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        device = next(self.parameters()).device
        prompt_ids = prompt_ids.to(device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        ctx = self.ctx_encoder(prompt_ids, pad_id=self.vocab.pad_id).unsqueeze(1)

        max_steps = max_bytes or self.config.max_bytes
        generated: list[int] = [self.vocab.bos_id]
        prefix = ""
        for _ in range(max_steps + 2):
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            pos = torch.arange(len(generated), device=device).unsqueeze(0)
            x = self.tok(input_ids) + self.pos(pos)
            pad_mask = input_ids.eq(self.vocab.pad_id)
            for layer in self.layers:
                x = layer(x, self_pad_mask=pad_mask, ctx=ctx)
            logits = self.lm_head(self.norm(x))[:, -1, :]  # [1, V]

            allowed = constraint.allowed_next(prefix)
            if not allowed:
                return None
            mask = torch.full_like(logits, float("-inf"))
            mask[0, list(allowed)] = 0.0
            logits = logits + mask

            if temperature <= 0.0 or top_k == 1:
                next_id = int(logits.argmax(dim=-1).item())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                topk = torch.topk(probs, min(top_k, probs.size(-1)))
                next_id = int(torch.multinomial(topk.values, num_samples=1).item())
                next_id = int(topk.indices[0, next_id].item())

            if next_id == self.vocab.eos_id:
                if constraint.is_complete(prefix):
                    return prefix
                return None
            generated.append(next_id)
            prefix = self.vocab.decode(generated, skip_special=True)
            if len(prefix.encode("utf-8")) > max_steps:
                return None
        return None

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> SurfaceAutoregressor:
        path = Path(path)
        payload = torch.load(path, map_location=device, weights_only=False)
        config = SurfaceAutoregressorConfig.from_dict(payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return model

    @classmethod
    def from_records(
        cls,
        records: Sequence[Any],
        *,
        config: SurfaceAutoregressorConfig | None = None,
        device: str = "cpu",
    ) -> SurfaceAutoregressor:
        """Build an untrained model; records are accepted for API symmetry."""
        del records
        model = cls(config or SurfaceAutoregressorConfig())
        model.to(device)
        return model


def train_surface_autoregressor(
    model: SurfaceAutoregressor,
    examples: list[tuple[str, str]],
    *,
    steps: int = 100,
    lr: float = 3e-3,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, float]:
    """Tiny fixture trainer: overfit the model on (prompt, target) pairs.

    Each example is a short prompt string and a target surface value. The model
    is trained teacher-forced with cross-entropy. Returns final loss.
    """
    import random

    torch.manual_seed(seed)
    random.seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vocab = model.vocab

    def make_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt, target = random.choice(examples)
        prompt_ids = torch.tensor(vocab.encode(prompt, add_special=True), dtype=torch.long)
        target_ids = vocab.encode(target, add_special=False)
        input_ids = [vocab.bos_id] + target_ids
        targets = target_ids + [vocab.eos_id]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            prompt_ids,
            torch.tensor(targets, dtype=torch.long),
        )

    losses: list[float] = []
    for _ in range(steps):
        input_ids, prompt_ids, targets = make_batch()
        input_ids = input_ids.unsqueeze(0).to(device)
        prompt_ids = prompt_ids.unsqueeze(0).to(device)
        targets = targets.unsqueeze(0).to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(input_ids, prompt_ids, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return {"final_loss": losses[-1], "initial_loss": losses[0]}


__all__ = [
    "DecorativeConstraint",
    "IdentifierConstraint",
    "SurfaceAutoregressor",
    "SurfaceAutoregressorConfig",
    "SurfaceByteVocab",
    "train_surface_autoregressor",
]
