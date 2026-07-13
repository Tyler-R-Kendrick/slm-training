"""TwoTower OpenUI model: frozen context encoder + trainable masked denoiser."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.blocks import DenoiserTower, TokenEncoder
from slm_training.models.tokenizer import OpenUITokenizer


@dataclass
class TwoTowerConfig:
    d_model: int = 128
    n_heads: int = 4
    context_layers: int = 2
    denoiser_layers: int = 4
    max_prompt_len: int = 128
    max_target_len: int = 256
    dropout: float = 0.0
    mask_min: float = 0.15
    mask_max: float = 0.85
    gen_steps: int = 8
    # False for from-scratch POC; True once context tower is a pretrained HF encoder
    freeze_context: bool = False
    seed: int = 0


def _pad_batch(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


class TwoTowerModel(nn.Module):
    """MaskGIT-style discrete diffusion conditioned on a prompt encoder."""

    def __init__(
        self,
        tokenizer: OpenUITokenizer,
        config: TwoTowerConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config or TwoTowerConfig()
        self.device_name = str(device)
        self.context = TokenEncoder(
            vocab_size=tokenizer.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.context_layers,
            n_heads=self.config.n_heads,
            max_len=self.config.max_prompt_len,
            dropout=self.config.dropout,
        )
        self.denoiser = DenoiserTower(
            vocab_size=tokenizer.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.denoiser_layers,
            n_heads=self.config.n_heads,
            max_len=self.config.max_target_len,
            dropout=self.config.dropout,
        )
        if self.config.freeze_context:
            for p in self.context.parameters():
                p.requires_grad = False
        self._rng = random.Random(self.config.seed)
        # Preferred decode length (fit to train corpus when available)
        self.gen_len = self.config.max_target_len
        self.to(device)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def _encode_context(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [
            self.tokenizer.encode(p)[: self.config.max_prompt_len] for p in prompts
        ]
        batch = _pad_batch(ids, self.tokenizer.pad_id).to(self.device_name)
        pad_mask = batch.eq(self.tokenizer.pad_id)
        with torch.set_grad_enabled(not self.config.freeze_context and self.training):
            ctx = self.context(batch, pad_id=self.tokenizer.pad_id)
        return ctx, pad_mask

    def _mask_targets(
        self, target_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return noisy_ids and boolean mask of positions to predict."""
        bsz, seq = target_ids.shape
        device = target_ids.device
        # Keep BOS/PAD fixed; allow EOS so the model learns where programs end.
        frozen = {self.tokenizer.pad_id, self.tokenizer.bos_id}
        noise = torch.zeros(bsz, seq, dtype=torch.bool, device=device)
        for i in range(bsz):
            valid = [
                j
                for j in range(seq)
                if int(target_ids[i, j]) not in frozen
            ]
            if not valid:
                continue
            rate = self._rng.uniform(self.config.mask_min, self.config.mask_max)
            k = max(1, int(math.ceil(rate * len(valid))))
            chosen = self._rng.sample(valid, k=min(k, len(valid)))
            noise[i, chosen] = True
        noisy = target_ids.clone()
        noisy[noise] = self.tokenizer.mask_id
        return noisy, noise

    def forward(self, batch: list[ExampleRecord]) -> float:
        """Train step: return scalar loss (also used by ModelPlugin)."""
        self.train()
        prompts = [r.prompt for r in batch]
        targets = [
            self.tokenizer.encode(r.openui)[: self.config.max_target_len]
            for r in batch
        ]
        target_ids = _pad_batch(targets, self.tokenizer.pad_id).to(self.device_name)
        ctx, ctx_pad = self._encode_context(prompts)
        noisy, predict_mask = self._mask_targets(target_ids)
        logits = self.denoiser(
            noisy, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
        )
        if not predict_mask.any():
            return 0.0
        loss = F.cross_entropy(
            logits[predict_mask],
            target_ids[predict_mask],
        )
        return float(loss.detach().cpu())

    def training_loss(self, batch: list[ExampleRecord]) -> torch.Tensor:
        """Differentiable loss tensor for optimizer step."""
        self.train()
        prompts = [r.prompt for r in batch]
        targets = [
            self.tokenizer.encode(r.openui)[: self.config.max_target_len]
            for r in batch
        ]
        target_ids = _pad_batch(targets, self.tokenizer.pad_id).to(self.device_name)
        ctx, ctx_pad = self._encode_context(prompts)
        noisy, predict_mask = self._mask_targets(target_ids)
        logits = self.denoiser(
            noisy, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
        )
        if not predict_mask.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[predict_mask], target_ids[predict_mask])

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        gold: ExampleRecord | None = None,
        max_len: int | None = None,
    ) -> str:
        """Iterative MaskGIT-style unmasking. Ignores gold content (no oracle)."""
        _ = gold
        self.eval()
        length = max_len or self.gen_len or self.config.max_target_len
        length = max(8, min(int(length), self.config.max_target_len))

        device = self.device_name
        ctx, ctx_pad = self._encode_context([prompt])
        ids = torch.full(
            (1, length), self.tokenizer.mask_id, dtype=torch.long, device=device
        )
        ids[0, 0] = self.tokenizer.bos_id
        unknown = ids.eq(self.tokenizer.mask_id)

        steps = max(1, self.config.gen_steps)
        for step in range(steps):
            if not unknown.any():
                break
            logits = self.denoiser(
                ids, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            conf = conf.masked_fill(~unknown, -1.0)
            remaining = int(unknown.sum().item())
            n_unmask = max(1, math.ceil(remaining / (steps - step)))
            flat_idx = conf.view(-1).topk(min(n_unmask, remaining)).indices
            for idx in flat_idx.tolist():
                b = idx // length
                t = idx % length
                if unknown[b, t]:
                    ids[b, t] = pred[b, t]
                    unknown[b, t] = False
            # Once EOS is known, freeze the suffix so trailing junk is not decoded.
            for b in range(ids.size(0)):
                eos_positions = (ids[b] == self.tokenizer.eos_id).nonzero(as_tuple=False)
                if eos_positions.numel() == 0:
                    continue
                end = int(eos_positions[0].item())
                if end + 1 < length:
                    ids[b, end + 1 :] = self.tokenizer.pad_id
                    unknown[b, end + 1 :] = False

        if unknown.any():
            logits = self.denoiser(
                ids, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
            pred = logits.argmax(dim=-1)
            ids[unknown] = pred[unknown]
            for b in range(ids.size(0)):
                eos_positions = (ids[b] == self.tokenizer.eos_id).nonzero(as_tuple=False)
                if eos_positions.numel() == 0:
                    continue
                end = int(eos_positions[0].item())
                if end + 1 < length:
                    ids[b, end + 1 :] = self.tokenizer.pad_id

        # Truncate at first EOS after BOS when present
        token_ids = ids[0].tolist()
        if self.tokenizer.eos_id in token_ids[1:]:
            end = token_ids.index(self.tokenizer.eos_id, 1)
            token_ids = token_ids[: end + 1]
        return self.tokenizer.decode(token_ids).strip()

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "twotower",
            "config": asdict(self.config),
            "gen_len": self.gen_len,
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
        }
        # Save tokenizer alongside
        tok_path = path.with_suffix(".tokenizer.json")
        self.tokenizer.save(tok_path)
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "kind": "twotower",
                    "config": asdict(self.config),
                    "gen_len": self.gen_len,
                    "tokenizer": str(tok_path.name),
                    "vocab_size": self.tokenizer.vocab_size,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        torch.save(payload, path)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        payload = torch.load(path, map_location=self.device_name, weights_only=False)
        if payload.get("kind") != "twotower":
            raise ValueError(f"checkpoint kind {payload.get('kind')!r} is not twotower")
        self.load_state_dict(payload["state_dict"])
        if "gen_len" in payload:
            self.gen_len = int(payload["gen_len"])
        tok_path = path.with_suffix(".tokenizer.json")
        if tok_path.exists():
            self.tokenizer = OpenUITokenizer.load(tok_path)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        device: str | torch.device = "cpu",
    ) -> TwoTowerModel:
        path = Path(path)
        payload = torch.load(path, map_location=device, weights_only=False)
        tok_path = path.with_suffix(".tokenizer.json")
        if not tok_path.exists():
            raise FileNotFoundError(f"missing tokenizer next to checkpoint: {tok_path}")
        tokenizer = OpenUITokenizer.load(tok_path)
        cfg = TwoTowerConfig(**payload.get("config") or {})
        model = cls(tokenizer=tokenizer, config=cfg, device=device)
        model.load_state_dict(payload["state_dict"])
        if "gen_len" in payload:
            model.gen_len = int(payload["gen_len"])
        return model

    @classmethod
    def from_records(
        cls,
        records: list[ExampleRecord],
        config: TwoTowerConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> TwoTowerModel:
        texts = [r.prompt for r in records] + [r.openui for r in records]
        tokenizer = OpenUITokenizer.build(texts)
        cfg = config or TwoTowerConfig()
        max_prompt = max(
            (len(tokenizer.encode(r.prompt)) for r in records), default=16
        )
        max_target = max(
            (len(tokenizer.encode(r.openui)) for r in records), default=32
        )
        # Keep embedding tables large enough for the corpus
        cfg.max_prompt_len = max(cfg.max_prompt_len, max_prompt + 4)
        cfg.max_target_len = max(cfg.max_target_len, max_target + 8)
        model = cls(tokenizer=tokenizer, config=cfg, device=device)
        # Decode at corpus length so generation does not spill into empty pad space
        model.gen_len = max(max_target + 2, 16)
        return model
