"""TwoTower OpenUI model: context encoder + trainable masked denoiser."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.blocks import DenoiserTower
from slm_training.models.context import (
    HFContextEncoder,
    ScratchContextEncoder,
    build_context_encoder,
    is_hf_context,
)
from slm_training.models.grammar import (
    apply_structural_bias,
    filter_ids_by_stream,
    pick_constrained_token,
    stream_check,
)
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
    # scratch | hf
    context_backend: str = "scratch"
    # Default production HF tower; tests may override with a tiny model.
    hf_model_name: str = "HuggingFaceTB/SmolLM2-135M"
    # True when using a pretrained HF context tower; optional for scratch.
    freeze_context: bool = False
    local_files_only: bool = False
    grammar_constrained: bool = True
    grammar_top_k: int = 16
    structural_bias: float = 1.25
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
        backend = (self.config.context_backend or "scratch").lower()
        freeze = self.config.freeze_context
        if backend in {"hf", "huggingface", "transformers"} and not freeze:
            # Explicit unfreeze allowed; factory typically sets freeze_context=True for HF.
            freeze = False

        self.context = build_context_encoder(
            backend=backend,
            vocab_size=tokenizer.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.context_layers,
            n_heads=self.config.n_heads,
            max_len=self.config.max_prompt_len,
            dropout=self.config.dropout,
            freeze=freeze,
            hf_model_name=self.config.hf_model_name,
            local_files_only=self.config.local_files_only,
        )
        self.denoiser = DenoiserTower(
            vocab_size=tokenizer.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.denoiser_layers,
            n_heads=self.config.n_heads,
            max_len=self.config.max_target_len,
            dropout=self.config.dropout,
        )
        self._rng = random.Random(self.config.seed)
        self.gen_len = self.config.max_target_len
        self.to(device)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def _encode_context(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if is_hf_context(self.context):
            assert isinstance(self.context, HFContextEncoder)
            # HFContextEncoder freezes the backbone but keeps proj trainable.
            return self.context.forward_prompts(
                prompts,
                max_len=self.config.max_prompt_len,
                device=self.device_name,
            )
        assert isinstance(self.context, ScratchContextEncoder)
        enable_grad = (not self.config.freeze_context) and self.training
        with torch.set_grad_enabled(enable_grad):
            return self.context.forward_prompts(
                prompts,
                encode_fn=self.tokenizer.encode,
                max_len=self.config.max_prompt_len,
                pad_id=self.tokenizer.pad_id,
                device=self.device_name,
            )

    def _mask_targets(
        self, target_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return noisy_ids and boolean mask of positions to predict."""
        bsz, seq = target_ids.shape
        device = target_ids.device
        frozen = {self.tokenizer.pad_id, self.tokenizer.bos_id}
        noise = torch.zeros(bsz, seq, dtype=torch.bool, device=device)
        for i in range(bsz):
            valid = [
                j for j in range(seq) if int(target_ids[i, j]) not in frozen
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
        self.train()
        loss = self.training_loss(batch)
        return float(loss.detach().cpu())

    def training_loss(self, batch: list[ExampleRecord]) -> torch.Tensor:
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

    def _decode_ids(self, ids_1d: torch.Tensor) -> str:
        token_ids = ids_1d.tolist()
        if self.tokenizer.eos_id in token_ids[1:]:
            end = token_ids.index(self.tokenizer.eos_id, 1)
            token_ids = token_ids[: end + 1]
        return self.tokenizer.decode(token_ids).strip()

    def _constrained_ltr_repair(
        self,
        ids: torch.Tensor,
        unknown: torch.Tensor,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Fill remaining masks left-to-right with streaming-parser filtering."""
        length = ids.size(1)
        for t in range(length):
            if not bool(unknown[0, t].item()):
                continue
            logits = self.denoiser(
                ids, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
            if self.config.structural_bias:
                logits = apply_structural_bias(
                    logits,
                    self.tokenizer,
                    bias=self.config.structural_bias,
                )
            prefix = ids[0, :t].tolist()
            choice = pick_constrained_token(
                logits[0, t],
                self.tokenizer,
                prefix,
                top_k=self.config.grammar_top_k,
            )
            ids[0, t] = choice
            unknown[0, t] = False
            if choice == self.tokenizer.eos_id:
                if t + 1 < length:
                    ids[0, t + 1 :] = self.tokenizer.pad_id
                    unknown[0, t + 1 :] = False
                break
        return ids

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        gold: ExampleRecord | None = None,
        max_len: int | None = None,
        grammar_constrained: bool | None = None,
    ) -> str:
        """Iterative MaskGIT-style unmasking with optional grammar constraints."""
        _ = gold
        self.eval()
        use_grammar = (
            self.config.grammar_constrained
            if grammar_constrained is None
            else grammar_constrained
        )
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
            if use_grammar and self.config.structural_bias:
                logits = apply_structural_bias(
                    logits,
                    self.tokenizer,
                    bias=self.config.structural_bias,
                )
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            conf = conf.masked_fill(~unknown, -1.0)
            remaining = int(unknown.sum().item())
            n_unmask = max(1, math.ceil(remaining / (steps - step)))
            flat_idx = conf.view(-1).topk(min(n_unmask, remaining)).indices
            newly: list[int] = []
            for idx in flat_idx.tolist():
                b = idx // length
                t = idx % length
                if unknown[b, t]:
                    ids[b, t] = pred[b, t]
                    unknown[b, t] = False
                    if b == 0:
                        newly.append(t)

            # Freeze suffix after EOS.
            for b in range(ids.size(0)):
                eos_positions = (ids[b] == self.tokenizer.eos_id).nonzero(
                    as_tuple=False
                )
                if eos_positions.numel() == 0:
                    continue
                end = int(eos_positions[0].item())
                if end + 1 < length:
                    ids[b, end + 1 :] = self.tokenizer.pad_id
                    unknown[b, end + 1 :] = False

            if use_grammar and newly:
                remask = filter_ids_by_stream(
                    self.tokenizer, ids[0].tolist(), newly
                )
                for t in remask:
                    ids[0, t] = self.tokenizer.mask_id
                    unknown[0, t] = True

        if unknown.any():
            if use_grammar:
                ids = self._constrained_ltr_repair(ids, unknown, ctx, ctx_pad)
            else:
                logits = self.denoiser(
                    ids, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
                )
                pred = logits.argmax(dim=-1)
                ids[unknown] = pred[unknown]
                for b in range(ids.size(0)):
                    eos_positions = (ids[b] == self.tokenizer.eos_id).nonzero(
                        as_tuple=False
                    )
                    if eos_positions.numel() == 0:
                        continue
                    end = int(eos_positions[0].item())
                    if end + 1 < length:
                        ids[b, end + 1 :] = self.tokenizer.pad_id

        text = self._decode_ids(ids[0])
        if use_grammar:
            try:
                status = stream_check(text)
                if status.serialized and status.complete_ok:
                    return status.serialized
            except Exception:  # noqa: BLE001
                pass
        return text

    def _state_dict_for_checkpoint(self) -> dict:
        state = {k: v.cpu() for k, v in self.state_dict().items()}
        # Keep checkpoints small: reload frozen HF backbone from hub/cache on load.
        if is_hf_context(self.context) and self.config.freeze_context:
            state = {
                k: v
                for k, v in state.items()
                if not k.startswith("context.backbone.")
            }
        return state

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "twotower",
            "config": asdict(self.config),
            "gen_len": self.gen_len,
            "state_dict": self._state_dict_for_checkpoint(),
        }
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
                    "context_backend": self.config.context_backend,
                    "hf_model_name": self.config.hf_model_name
                    if is_hf_context(self.context)
                    else None,
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
        missing, unexpected = self.load_state_dict(
            payload["state_dict"], strict=False
        )
        _ = missing, unexpected
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
        raw_cfg = dict(payload.get("config") or {})
        # Ignore unknown keys for forward/back compat
        valid = {f.name for f in TwoTowerConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = TwoTowerConfig(**{k: v for k, v in raw_cfg.items() if k in valid})
        model = cls(tokenizer=tokenizer, config=cfg, device=device)
        model.load_state_dict(payload["state_dict"], strict=False)
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
        cfg.max_prompt_len = max(cfg.max_prompt_len, max_prompt + 4)
        cfg.max_target_len = max(cfg.max_target_len, max_target + 8)
        model = cls(tokenizer=tokenizer, config=cfg, device=device)
        model.gen_len = max(max_target + 2, 16)
        return model
