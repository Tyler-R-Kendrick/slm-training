"""Grammar-native block diffusion model over production + slot pointers."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.models.block_noise import BlockNoiseSchedule, corrupt_blocks_for_training
from slm_training.models.blocks import DenoiserTower
from slm_training.models.constrained_posterior import (
    ExtendabilityChecker,
    adaptive_should_stop,
    apply_commits,
    parallel_commit_selection,
    pick_constrained_production,
)
from slm_training.models.context import ScratchContextEncoder, build_context_encoder, is_hf_context
from slm_training.models.tokenizer import OpenUITokenizer, tokenize_text
from slm_training.models.twotower import format_context_text


def _load_production_codec(texts: list[str]):
    from slm_training.dsl.production_codec import ProductionCodec

    return ProductionCodec.build(texts)


def _codec_kind_from_vocab(id_to_production: dict[int, str]) -> str:
    for tok in id_to_production.values():
        if tok.startswith(("+", "@", "^", "&", "#")) or tok in {"=", "-", "[", "]"}:
            return "production"
        if tok in {"NEWLINE", "ASSIGN", "SLOT", "LPAREN", "RPAREN"} or tok.startswith(
            ("COMP:", "VAR:", "LIT:", "REF:", "TOK:")
        ):
            return "inline"
    return "production"


def _restore_codec(raw_codec: dict[str, Any]) -> InlineProductionCodec:
    id_to_prod = {
        int(k): v for k, v in (raw_codec.get("id_to_production") or {}).items()
    }
    kind = str(raw_codec.get("codec_kind") or _codec_kind_from_vocab(id_to_prod))
    common = dict(
        production_to_id=dict(raw_codec.get("production_to_id") or {}),
        id_to_production=id_to_prod,
        pad_id=int(raw_codec.get("pad_id", 0)),
        bos_id=int(raw_codec.get("bos_id", 1)),
        eos_id=int(raw_codec.get("eos_id", 2)),
        mask_id=int(raw_codec.get("mask_id", 3)),
        slot_none_id=int(raw_codec.get("slot_none_id", 0)),
    )
    if kind == "production":
        from slm_training.dsl.production_codec import ProductionCodec

        return ProductionCodec(
            **common,
            unk_id=int(raw_codec.get("unk_id", 4)),
        )
    return InlineProductionCodec(**common)


@dataclass
class InlineProductionCodec:
    """Minimal production linearization until dsl.production_codec ships."""

    production_to_id: dict[str, int] = field(default_factory=dict)
    id_to_production: dict[int, str] = field(default_factory=dict)
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    mask_id: int = 3
    slot_none_id: int = 0

    @classmethod
    def build(cls, texts: list[str]) -> InlineProductionCodec:
        specials = ["<pad>", "<bos>", "<eos>", "<mask>"]
        vocab: dict[str, int] = {tok: i for i, tok in enumerate(specials)}
        structural = ["NEWLINE", "ASSIGN", "LPAREN", "RPAREN", "LBRACK", "RBRACK", "COMMA", "SPACE", "SLOT"]
        for tok in structural:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        for text in texts:
            for prod in cls._extract_production_names(text):
                if prod not in vocab:
                    vocab[prod] = len(vocab)
        inv = {i: t for t, i in vocab.items()}
        return cls(
            production_to_id=vocab,
            id_to_production=inv,
            pad_id=vocab["<pad>"],
            bos_id=vocab["<bos>"],
            eos_id=vocab["<eos>"],
            mask_id=vocab["<mask>"],
        )

    @staticmethod
    def _extract_production_names(text: str) -> list[str]:
        names: list[str] = []
        for prod, _slot in InlineProductionCodec._walk(text, ["__noop__"]):
            names.append(prod)
        return names

    @staticmethod
    def _walk(text: str, slot_inventory: list[str]) -> list[tuple[str, int]]:
        inv = {ph if ph.startswith(":") else f":{ph}": i + 1 for i, ph in enumerate(slot_inventory)}
        tokens = tokenize_text(text)
        out: list[tuple[str, int]] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "\n":
                out.append(("NEWLINE", 0))
                i += 1
                continue
            if tok == " ":
                out.append(("SPACE", 0))
                i += 1
                continue
            if tok in {"=", "(", ")", "[", "]", ","}:
                mapping = {
                    "=": "ASSIGN",
                    "(": "LPAREN",
                    ")": "RPAREN",
                    "[": "LBRACK",
                    "]": "RBRACK",
                    ",": "COMMA",
                }
                out.append((mapping[tok], 0))
                i += 1
                continue
            if tok == '"':
                placeholder = InlineProductionCodec._read_placeholder(tokens, i)
                if placeholder is not None:
                    ph, end = placeholder
                    slot_id = inv.get(ph, 0)
                    out.append(("SLOT", slot_id))
                    i = end + 1
                    continue
                literal, end = InlineProductionCodec._read_literal(tokens, i)
                out.append((f'LIT:{literal}', 0))
                i = end + 1
                continue
            if i + 1 < len(tokens) and tokens[i + 1] == "=":
                out.append((f"VAR:{tok}", 0))
                i += 1
                continue
            if tok and tok[0].isupper() and tok.isidentifier():
                out.append((f"COMP:{tok}", 0))
                i += 1
                continue
            if tok.isidentifier():
                out.append((f"REF:{tok}", 0))
                i += 1
                continue
            out.append((f"TOK:{tok}", 0))
            i += 1
        return out

    @staticmethod
    def _read_placeholder(tokens: list[str], start: int) -> tuple[str, int] | None:
        if start >= len(tokens) or tokens[start] != '"':
            return None
        if start + 1 >= len(tokens) or tokens[start + 1] != ":":
            return None
        parts = [":"]
        i = start + 2
        while i < len(tokens):
            if tokens[i] == '"':
                ph = "".join(parts)
                return ph, i
            parts.append(tokens[i])
            i += 1
        return None

    @staticmethod
    def _read_literal(tokens: list[str], start: int) -> tuple[str, int]:
        if start >= len(tokens) or tokens[start] != '"':
            return "", start
        parts = ['"']
        i = start + 1
        while i < len(tokens):
            parts.append(tokens[i])
            if tokens[i] == '"':
                return "".join(parts), i
            i += 1
        return "".join(parts), i

    @property
    def vocab_size(self) -> int:
        return len(self.production_to_id)

    def encode(
        self,
        openui: str,
        slot_inventory: list[str] | None = None,
        *,
        max_len: int = 256,
    ) -> tuple[list[int], list[int]]:
        inventory = list(slot_inventory or extract_placeholders(openui))
        steps = self._walk(openui, inventory)
        prod_ids = [self.bos_id]
        slot_ids = [self.slot_none_id]
        for prod, slot in steps:
            pid = self.production_to_id.get(prod)
            if pid is None:
                pid = self.production_to_id.setdefault(prod, len(self.production_to_id))
                self.id_to_production[pid] = prod
            prod_ids.append(pid)
            slot_ids.append(int(slot))
        prod_ids.append(self.eos_id)
        slot_ids.append(self.slot_none_id)
        prod_ids = prod_ids[:max_len]
        slot_ids = slot_ids[: len(prod_ids)]
        if len(slot_ids) < len(prod_ids):
            slot_ids.extend([self.slot_none_id] * (len(prod_ids) - len(slot_ids)))
        return prod_ids, slot_ids

    def decode(
        self,
        production_ids: list[int],
        slot_ids: list[int],
        slot_inventory: list[str],
        *,
        stop_at_mask: bool = False,
    ) -> str:
        pieces: list[str] = []
        for idx, (pid, sid) in enumerate(zip(production_ids, slot_ids)):
            if stop_at_mask and pid == self.mask_id:
                pieces.append("<mask>")
                break
            if pid in {self.pad_id, self.bos_id}:
                continue
            if pid == self.eos_id:
                break
            if pid == self.mask_id:
                pieces.append("<mask>")
                continue
            prod = self.id_to_production.get(pid, "")
            if prod == "NEWLINE":
                if pieces and not pieces[-1].endswith("\n"):
                    pieces.append("\n")
            elif prod == "SPACE":
                if pieces and not pieces[-1].endswith((" ", "\n", "(", "[", ",")):
                    pieces.append(" ")
            elif prod == "ASSIGN":
                pieces.append(" = ")
            elif prod == "LPAREN":
                pieces.append("(")
            elif prod == "RPAREN":
                pieces.append(")")
            elif prod == "LBRACK":
                pieces.append("[")
            elif prod == "RBRACK":
                pieces.append("]")
            elif prod == "COMMA":
                pieces.append(", ")
            elif prod == "SLOT":
                slot_idx = int(sid) - 1
                if 0 <= slot_idx < len(slot_inventory):
                    ph = slot_inventory[slot_idx]
                    if not ph.startswith(":"):
                        ph = f":{ph}"
                    pieces.append(f'"{ph}"')
                else:
                    pieces.append('":slot.unknown"')
            elif prod.startswith("VAR:"):
                name = prod.split(":", 1)[1]
                if pieces and not pieces[-1].endswith(("\n", " ")):
                    pieces.append(" ")
                pieces.append(name)
            elif prod.startswith("COMP:"):
                name = prod.split(":", 1)[1]
                pieces.append(name)
            elif prod.startswith("REF:"):
                name = prod.split(":", 1)[1]
                pieces.append(name)
            elif prod.startswith("LIT:"):
                pieces.append(prod.split(":", 1)[1])
            elif prod.startswith("TOK:"):
                pieces.append(prod.split(":", 1)[1])
            _ = idx
        text = "".join(pieces)
        return text.replace("  ", " ").replace(" \n", "\n").strip() + ("\n" if text.endswith("\n") else "")


@dataclass
class GrammarDiffusionConfig:
    d_model: int = 96
    n_heads: int = 4
    context_layers: int = 2
    denoiser_layers: int = 3
    max_prompt_len: int = 192
    max_target_len: int = 192
    dropout: float = 0.0
    block_size: int = 4
    mask_min: float = 0.15
    mask_max: float = 0.85
    gen_steps: int = 8
    max_slots: int = 16
    context_backend: str = "scratch"
    hf_model_name: str = "HuggingFaceTB/SmolLM2-135M"
    freeze_context: bool = False
    local_files_only: bool = False
    grammar_dsl: str = "openui"
    parallel_unmask: str = "adaptive"
    grammar_top_k: int = 8
    production_loss_weight: float = 1.0
    slot_loss_weight: float = 0.5
    confidence_loss_weight: float = 0.25
    design_md_in_context: bool = False
    design_md_budget: int = 1200
    schema_in_context: bool = False
    slot_contract_in_context: bool = True
    slot_contract_constrained_decode: bool = True
    # E54/E35: derive inventory from prompt/DESIGN.md only (never gold.placeholders).
    honest_slot_contract: bool = True
    seed: int = 0
    eval_mode_no_fallback: bool = True

    @property
    def block_schedule(self) -> BlockNoiseSchedule:
        return BlockNoiseSchedule(
            block_size=self.block_size,
            mask_min=self.mask_min,
            mask_max=self.mask_max,
            gen_steps=self.gen_steps,
        )


class GrammarDenoiser(nn.Module):
    """Masked production denoiser with slot-pointer and confidence heads."""

    def __init__(
        self,
        n_productions: int,
        max_slots: int,
        d_model: int = 96,
        n_layers: int = 3,
        n_heads: int = 4,
        max_len: int = 192,
        dropout: float = 0.0,
        pad_id: int = 0,
        mask_id: int = 3,
    ) -> None:
        super().__init__()
        self.core = DenoiserTower(
            vocab_size=n_productions,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
            dropout=dropout,
        )
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.slot_head = nn.Linear(d_model, max_slots + 1)
        self.confidence_head = nn.Linear(d_model, 1)

    def forward(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        *,
        ctx_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self._hidden(noisy_ids, context, ctx_pad_mask=ctx_pad_mask)
        production_logits = self.core.lm_head(self.core.norm(hidden))
        slot_logits = self.slot_head(hidden)
        confidence = torch.sigmoid(self.confidence_head(hidden).squeeze(-1))
        return production_logits, slot_logits, confidence

    def _hidden(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        *,
        ctx_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seq = noisy_ids.shape
        if seq > self.core.max_len:
            noisy_ids = noisy_ids[:, : self.core.max_len]
            seq = self.core.max_len
        pos = torch.arange(seq, device=noisy_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.core.tok(noisy_ids) + self.core.pos(pos)
        self_pad = noisy_ids.eq(self.pad_id)
        for layer in self.core.layers:
            x = layer(x, self_pad_mask=self_pad, ctx=context, ctx_pad_mask=ctx_pad_mask)
        return x


def _pad_batch(seqs: list[list[int]], pad_id: int, device: str | torch.device) -> torch.Tensor:
    max_len = max((len(s) for s in seqs), default=1)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        if s:
            out[i, : len(s)] = torch.as_tensor(s, dtype=torch.long, device=device)
    return out


class GrammarDiffusionModel(nn.Module):
    """Block diffusion over grammar productions with constrained posterior decode."""

    def __init__(
        self,
        tokenizer: OpenUITokenizer,
        codec: InlineProductionCodec,
        config: GrammarDiffusionConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.codec = codec
        self.config = config or GrammarDiffusionConfig()
        self.device_name = str(device)
        backend = (self.config.context_backend or "scratch").lower()
        self.context = build_context_encoder(
            backend=backend,
            vocab_size=tokenizer.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.context_layers,
            n_heads=self.config.n_heads,
            max_len=self.config.max_prompt_len,
            dropout=self.config.dropout,
            freeze=self.config.freeze_context,
            hf_model_name=self.config.hf_model_name,
            local_files_only=self.config.local_files_only,
        )
        self.denoiser = GrammarDenoiser(
            n_productions=max(8, self.codec.vocab_size),
            max_slots=self.config.max_slots,
            d_model=self.config.d_model,
            n_layers=self.config.denoiser_layers,
            n_heads=self.config.n_heads,
            max_len=self.config.max_target_len,
            dropout=self.config.dropout,
            pad_id=self.codec.pad_id,
            mask_id=self.codec.mask_id,
        )
        self._rng = random.Random(self.config.seed)
        self._extend = ExtendabilityChecker(grammar_dsl=self.config.grammar_dsl)
        self.to(device)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def _encode_context(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if is_hf_context(self.context):
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

    def _format_context(
        self,
        prompt: str,
        *,
        design_md: str | None = None,
        slot_contract: list[str] | None = None,
        schema: str | None = None,
    ) -> str:
        dm = design_md if self.config.design_md_in_context else None
        contract = slot_contract if self.config.slot_contract_in_context else None
        if schema is None and self.config.schema_in_context:
            from slm_training.harnesses.quality import compact_schema_snippet

            schema = compact_schema_snippet(budget=min(600, self.config.design_md_budget))
        return format_context_text(
            prompt,
            dm,
            budget=self.config.design_md_budget,
            schema=schema,
            slot_contract=contract,
        )

    def forward(self, batch: list[ExampleRecord]) -> float:
        self.train()
        return float(self.training_loss(batch).detach().cpu())

    def training_loss(self, batch: list[ExampleRecord]) -> torch.Tensor:
        self.train()
        prompts: list[str] = []
        prod_targets: list[list[int]] = []
        slot_targets: list[list[int]] = []
        for record in batch:
            inventory = list(record.placeholders or extract_placeholders(record.openui))
            prompts.append(
                self._format_context(
                    record.prompt,
                    design_md=record.design_md,
                    slot_contract=inventory,
                )
            )
            prod, slot = self.codec.encode(
                record.openui,
                inventory,
                max_len=self.config.max_target_len,
            )
            prod_targets.append(prod)
            slot_targets.append(slot)

        prod_ids = _pad_batch(prod_targets, self.codec.pad_id, self.device_name)
        slot_ids = _pad_batch(slot_targets, self.codec.slot_none_id, self.device_name)
        ctx, ctx_pad = self._encode_context(prompts)

        frozen = prod_ids.eq(self.codec.pad_id) | prod_ids.eq(self.codec.bos_id)
        noisy, predict_mask = corrupt_blocks_for_training(
            prod_ids.size(1),
            schedule=self.config.block_schedule,
            mask_id=self.codec.mask_id,
            pad_id=self.codec.pad_id,
            frozen=frozen,
            target_ids=prod_ids,
        )

        prod_logits, slot_logits, confidence = self.denoiser(
            noisy, ctx, ctx_pad_mask=ctx_pad
        )

        total = prod_logits.sum() * 0.0
        if predict_mask.any():
            prod_ce = F.cross_entropy(
                prod_logits[predict_mask],
                prod_ids[predict_mask],
                reduction="mean",
            )
            total = total + self.config.production_loss_weight * prod_ce

            slot_mask = predict_mask & slot_ids.ne(self.codec.slot_none_id)
            if slot_mask.any():
                slot_ce = F.cross_entropy(
                    slot_logits[slot_mask],
                    slot_ids[slot_mask],
                    reduction="mean",
                )
                total = total + self.config.slot_loss_weight * slot_ce

            with torch.no_grad():
                pred_prod = prod_logits.argmax(dim=-1)
                correct = (pred_prod == prod_ids) & predict_mask
                if slot_mask.any():
                    pred_slot = slot_logits.argmax(dim=-1)
                    correct = correct & (~slot_mask | (pred_slot == slot_ids))
                target_conf = correct.float()
            conf_on_mask = confidence[predict_mask]
            target_on_mask = target_conf[predict_mask]
            if conf_on_mask.numel():
                calib = F.binary_cross_entropy(conf_on_mask, target_on_mask)
                total = total + self.config.confidence_loss_weight * calib

        return total

    @torch.inference_mode()
    def generate_batch_requests(self, requests: list[GenerationRequest]) -> list[str]:
        """Production-only inputs; constrained block diffusion without eval fallback."""
        self.eval()
        if not requests:
            return []
        prompts = [
            self._format_context(
                req.prompt,
                design_md=req.design_md,
                slot_contract=list(req.slot_contract) if req.slot_contract else None,
                schema=req.schema,
            )
            for req in requests
        ]
        ctx, ctx_pad = self._encode_context(prompts)
        out: list[str] = []
        for i, req in enumerate(requests):
            inventory = [
                ph if ph.startswith(":") else f":{ph}" for ph in (req.slot_contract or ())
            ]
            if not inventory:
                from slm_training.models.template_fill import inventory_from_prompt

                inventory = inventory_from_prompt(
                    req.prompt, req.design_md, heuristic=True
                )
            text = self._decode_one(
                ctx[i : i + 1],
                ctx_pad[i : i + 1],
                slot_inventory=inventory,
            )
            out.append(text)
        return out

    def generate(self, prompt: str, gold: ExampleRecord | None = None) -> str:
        """Generate without reading ``gold.placeholders`` (E35/E54 honesty).

        Inventory comes from the user-visible prompt / DESIGN.md only. ``gold``
        may supply ``design_md`` text for conditioning, never a hidden slot list.
        """
        from slm_training.models.template_fill import inventory_from_prompt

        design_md = gold.design_md if gold is not None else None
        honest = bool(getattr(self.config, "honest_slot_contract", True))
        if honest:
            contract = tuple(
                inventory_from_prompt(prompt, design_md, heuristic=True)
            )
        else:
            # Legacy escape hatch — still prefer prompt-visible inventory first.
            contract = tuple(
                inventory_from_prompt(prompt, design_md, heuristic=False)
            )
            if not contract and gold is not None and gold.placeholders:
                contract = tuple(gold.placeholders)
            if not contract:
                contract = tuple(
                    inventory_from_prompt(prompt, design_md, heuristic=True)
                )
        return self.generate_batch_requests(
            [
                GenerationRequest(
                    prompt=prompt,
                    slot_contract=contract,
                    design_md=design_md,
                )
            ]
        )[0]

    def _decode_one(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        *,
        slot_inventory: list[str],
    ) -> str:
        length = self.config.max_target_len
        device = self.device_name
        prod = torch.full((1, length), self.codec.mask_id, dtype=torch.long, device=device)
        slot = torch.zeros((1, length), dtype=torch.long, device=device)
        prod[0, 0] = self.codec.bos_id
        unknown = prod.eq(self.codec.mask_id)

        prod_list = prod[0].tolist()
        slot_list = slot[0].tolist()
        ltr_fill = type(self.codec).__name__ == "ProductionCodec"

        for step in range(self.config.gen_steps):
            if not bool(unknown.any()):
                break
            prod_logits, slot_logits, confidence = self.denoiser(
                prod, ctx, ctx_pad_mask=ctx_pad
            )
            if ltr_fill:
                commits = self._ltr_commits(
                    prod_logits,
                    slot_logits,
                    unknown,
                    prod_list=prod_list,
                    slot_list=slot_list,
                    slot_inventory=slot_inventory,
                )
            else:
                commits = parallel_commit_selection(
                    prod_logits,
                    slot_logits,
                    confidence,
                    unknown,
                    production_ids=prod_list,
                    slot_ids=slot_list,
                    slot_inventory=slot_inventory,
                    codec=self.codec,
                    checker=self._extend,
                    schedule=self.config.block_schedule,
                    step=step,
                    mode=self.config.parallel_unmask,
                    top_k=self.config.grammar_top_k,
                )
            if not commits:
                break
            apply_commits(prod, slot, unknown, commits)
            prod_list = prod[0].tolist()
            slot_list = slot[0].tolist()
            if adaptive_should_stop(unknown, confidence, step=step, schedule=self.config.block_schedule):
                break

        if bool(unknown.any()):
            prod_logits, slot_logits, confidence = self.denoiser(
                prod, ctx, ctx_pad_mask=ctx_pad
            )
            if ltr_fill:
                for _ in range(length):
                    if not bool(unknown.any()):
                        break
                    commits = self._ltr_commits(
                        prod_logits,
                        slot_logits,
                        unknown,
                        prod_list=prod_list,
                        slot_list=slot_list,
                        slot_inventory=slot_inventory,
                        block_size=1,
                    )
                    if not commits:
                        break
                    apply_commits(prod, slot, unknown, commits)
                    prod_list = prod[0].tolist()
                    slot_list = slot[0].tolist()
            else:
                for pos in range(length):
                    if not bool(unknown[0, pos].item()):
                        continue
                    commits = parallel_commit_selection(
                        prod_logits,
                        slot_logits,
                        confidence,
                        unknown,
                        production_ids=prod_list,
                        slot_ids=slot_list,
                        slot_inventory=slot_inventory,
                        codec=self.codec,
                        checker=self._extend,
                        schedule=BlockNoiseSchedule(
                            block_size=1,
                            mask_min=1.0,
                            mask_max=1.0,
                            gen_steps=1,
                        ),
                        step=0,
                        mode="topk",
                        top_k=self.config.grammar_top_k,
                    )
                    pos_commits = [c for c in commits if c[0] == pos]
                    if pos_commits:
                        apply_commits(prod, slot, unknown, pos_commits)
                        prod_list = prod[0].tolist()
                        slot_list = slot[0].tolist()

        decoded = self.codec.decode(prod_list, slot_list, slot_inventory)
        if self.config.eval_mode_no_fallback:
            return decoded.strip()
        return decoded.strip()

    def _ltr_commits(
        self,
        prod_logits: torch.Tensor,
        slot_logits: torch.Tensor,
        unknown: torch.Tensor,
        *,
        prod_list: list[int],
        slot_list: list[int],
        slot_inventory: list[str],
        block_size: int | None = None,
    ) -> list[tuple[int, int, int, float]]:
        """Left-to-right commits for grammar-native production streams (no holes)."""
        if not bool(unknown.any()):
            return []
        width = block_size or self.config.block_schedule.block_size
        leftmost = int(torch.where(unknown[0])[0][0].item())
        end = min(unknown.size(-1), leftmost + max(1, width))
        commits: list[tuple[int, int, int, float]] = []
        prod_canvas = list(prod_list)
        slot_canvas = list(slot_list)
        for pos in range(leftmost, end):
            if pos >= unknown.size(-1) or not bool(unknown[0, pos].item()):
                continue
            picked = pick_constrained_production(
                prod_logits,
                slot_logits,
                position=pos,
                production_ids=prod_canvas,
                slot_ids=slot_canvas,
                slot_inventory=slot_inventory,
                codec=self.codec,
                checker=self._extend,
                top_k=self.config.grammar_top_k,
                slot_none_id=self.codec.slot_none_id,
            )
            if picked is None:
                break
            prod_id, slot_id, conf = picked
            commits.append((pos, prod_id, slot_id, conf))
            prod_canvas[pos] = prod_id
            slot_canvas[pos] = slot_id
        return commits

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "kind": "grammar_diffusion",
            "config": asdict(self.config),
            "codec": {
                "codec_kind": (
                    "production"
                    if type(self.codec).__name__ == "ProductionCodec"
                    else "inline"
                ),
                "production_to_id": self.codec.production_to_id,
                "id_to_production": {str(k): v for k, v in self.codec.id_to_production.items()},
                "pad_id": self.codec.pad_id,
                "bos_id": self.codec.bos_id,
                "eos_id": self.codec.eos_id,
                "mask_id": self.codec.mask_id,
                "slot_none_id": self.codec.slot_none_id,
                "unk_id": getattr(self.codec, "unk_id", 4),
            },
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
        }
        tok_path = path.with_suffix(".tokenizer.json")
        self.tokenizer.save(tok_path)
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "kind": "grammar_diffusion",
                    "tokenizer": str(tok_path.name),
                    "vocab_size": self.tokenizer.vocab_size,
                    "production_vocab": self.codec.vocab_size,
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
        if payload.get("kind") != "grammar_diffusion":
            raise ValueError(
                f"checkpoint kind {payload.get('kind')!r} is not grammar_diffusion"
            )
        raw_codec = payload.get("codec") or {}
        self.codec = _restore_codec(raw_codec)
        self.load_state_dict(payload["state_dict"], strict=False)
        tok_path = path.with_suffix(".tokenizer.json")
        if tok_path.exists():
            self.tokenizer = OpenUITokenizer.load(tok_path)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        device: str | torch.device = "cpu",
    ) -> GrammarDiffusionModel:
        path = Path(path)
        payload = torch.load(path, map_location=device, weights_only=False)
        tok_path = path.with_suffix(".tokenizer.json")
        if not tok_path.exists():
            raise FileNotFoundError(f"missing tokenizer next to checkpoint: {tok_path}")
        tokenizer = OpenUITokenizer.load(tok_path)
        raw_cfg = dict(payload.get("config") or {})
        valid = {f.name for f in GrammarDiffusionConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = GrammarDiffusionConfig(**{k: v for k, v in raw_cfg.items() if k in valid})
        raw_codec = payload.get("codec") or {}
        codec = _restore_codec(raw_codec)
        model = cls(tokenizer=tokenizer, codec=codec, config=cfg, device=device)
        model.load_state_dict(payload["state_dict"], strict=False)
        return model

    @classmethod
    def from_records(
        cls,
        records: list[ExampleRecord],
        config: GrammarDiffusionConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> GrammarDiffusionModel:
        texts = [r.openui for r in records]
        codec = _load_production_codec(texts)
        prompt_texts = [r.prompt for r in records]
        tokenizer = OpenUITokenizer.build(prompt_texts + texts)
        cfg = config or GrammarDiffusionConfig()
        max_prompt = max((len(tokenizer.encode(r.prompt)) for r in records), default=16)
        max_prod = max(
            len(codec.encode(r.openui, list(r.placeholders or extract_placeholders(r.openui)))[0])
            for r in records
        ) if records else 32
        cfg.max_prompt_len = max(cfg.max_prompt_len, max_prompt + 4)
        cfg.max_target_len = max(cfg.max_target_len, max_prod + 4)
        return cls(tokenizer=tokenizer, codec=codec, config=cfg, device=device)


__all__ = [
    "GrammarDiffusionConfig",
    "GrammarDiffusionModel",
    "GrammarDenoiser",
    "InlineProductionCodec",
]
