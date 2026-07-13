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
from slm_training.models.parallel_decode import select_unmask_indices
from slm_training.models.tokenizer import OpenUITokenizer


def format_context_text(
    prompt: str,
    design_md: str | None = None,
    *,
    budget: int = 1800,
    schema: str | None = None,
    retrieved_skeleton: str | None = None,
) -> str:
    """Concatenate prompt with optional schema / skeleton / DESIGN.md."""
    prompt = (prompt or "").strip()
    parts = [prompt] if prompt else []
    if schema and schema.strip():
        parts.append(f"---SCHEMA---\n{schema.strip()[: min(600, budget)]}")
    if retrieved_skeleton and retrieved_skeleton.strip():
        parts.append(
            f"---RETRIEVED_SKELETON---\n{retrieved_skeleton.strip()[: min(400, budget)]}"
        )
    if design_md and design_md.strip():
        dm = design_md.strip()
        if len(dm) > budget:
            dm = dm[:budget].rsplit("\n", 1)[0]
        parts.append(f"---DESIGN.md---\n{dm}")
    return "\n\n".join(parts) if parts else prompt


@dataclass
class TwoTowerConfig:
    d_model: int = 128
    n_heads: int = 4
    context_layers: int = 2
    denoiser_layers: int = 4
    max_prompt_len: int = 256
    max_target_len: int = 256
    dropout: float = 0.0
    mask_min: float = 0.15
    mask_max: float = 0.85
    gen_steps: int = 8
    # scratch | hf — ModelBuildConfig / CLI default to hf for production runs.
    context_backend: str = "scratch"
    # Default production HF tower; tests may override with a tiny model.
    hf_model_name: str = "HuggingFaceTB/SmolLM2-135M"
    # True when using a pretrained HF context tower; optional for scratch.
    freeze_context: bool = False
    local_files_only: bool = False
    grammar_constrained: bool = True
    grammar_top_k: int = 16
    structural_bias: float = 1.25
    # Full LTR constrained repair is accurate but slow (Node stream_check per token).
    # Off by default; enable for final quality evals.
    grammar_ltr_repair: bool = False
    grammar_ltr_max_tokens: int = 64
    # Progressive LTR canvases (short first). Typical programs finish in the
    # first stage so we avoid O(T²) cost of a full max-length canvas.
    grammar_ltr_stages: tuple[int, ...] = (32, 48, 96)
    # Finalize LTR text with Node validate (adds ~1–2ms). Off by default —
    # eval already validates via meaningful-parse.
    grammar_finalize_validate: bool = False
    # Eval / throughput: batch size for generate_batch.
    generate_batch_size: int = 16
    # When True and grammar_constrained, skip MaskGIT and decode LTR only.
    grammar_ltr_primary: bool = False
    # Mix teacher-forced next-token CE into training (helps LTR generate).
    ltr_loss_weight: float = 0.5
    # Extra CE weight on gold placeholder token positions (fidelity signal).
    fidelity_loss_weight: float = 0.0
    design_md_in_context: bool = True
    design_md_budget: int = 1800
    schema_in_context: bool = False
    retrieval_k: int = 0
    best_of_n: int = 1
    seed: int = 0
    # Accelerator / SOTA decode knobs
    use_compile: bool = False
    compile_mode: str = "default"
    use_amp: bool = False
    # MaskGIT parallel unmask: topk | confidence | adaptive (mean-field-lite)
    parallel_unmask: str = "adaptive"


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
        # Optional retrieval bank: list[(norm_prompt, openui, id)]
        self.skeleton_bank: list[tuple[str, str, str]] = []
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
        prompts = [
            self._format_one_context(r.prompt, r.design_md, query_prompt=r.prompt)
            for r in batch
        ]
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
        if predict_mask.any():
            mask_loss = F.cross_entropy(logits[predict_mask], target_ids[predict_mask])
        else:
            mask_loss = logits.sum() * 0.0

        fid_w = float(getattr(self.config, "fidelity_loss_weight", 0.0) or 0.0)
        if fid_w > 0.0 and predict_mask.any():
            # Upweight CE on gold placeholder string tokens (":ns.slot").
            ph_mask = torch.zeros_like(predict_mask)
            for i in range(target_ids.size(0)):
                for j in range(target_ids.size(1)):
                    if not bool(predict_mask[i, j].item()):
                        continue
                    tok = self.tokenizer.id_to_token.get(int(target_ids[i, j]), "")
                    if '":' in tok or (tok.startswith(":") and "." in tok):
                        ph_mask[i, j] = True
            if ph_mask.any():
                fid_loss = F.cross_entropy(logits[ph_mask], target_ids[ph_mask])
                mask_loss = mask_loss + fid_w * fid_loss

        ltr_w = float(self.config.ltr_loss_weight or 0.0)
        if ltr_w <= 0.0 or target_ids.size(1) < 2:
            return mask_loss

        # Prefix-LM style: mask a random suffix and predict those tokens.
        bsz, seq = target_ids.shape
        ltr_noisy = target_ids.clone()
        ltr_mask = torch.zeros_like(target_ids, dtype=torch.bool)
        for i in range(bsz):
            cut = self._rng.randint(1, max(1, seq - 1))
            ltr_noisy[i, cut:] = self.tokenizer.mask_id
            for j in range(cut, seq):
                if int(target_ids[i, j]) == self.tokenizer.pad_id:
                    break
                ltr_mask[i, j] = True
        ltr_logits = self.denoiser(
            ltr_noisy, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
        )
        if ltr_mask.any():
            ltr_loss = F.cross_entropy(ltr_logits[ltr_mask], target_ids[ltr_mask])
        else:
            ltr_loss = mask_loss * 0.0
        return mask_loss + ltr_w * ltr_loss

    def _format_one_context(
        self,
        prompt: str,
        design_md: str | None,
        *,
        query_prompt: str | None = None,
    ) -> str:
        schema = None
        if getattr(self.config, "schema_in_context", False):
            from slm_training.quality import compact_schema_snippet

            schema = compact_schema_snippet(budget=min(600, self.config.design_md_budget))
        skeleton = None
        k = int(getattr(self.config, "retrieval_k", 0) or 0)
        if k > 0 and self.skeleton_bank:
            from slm_training.retrieval import format_retrieved_skeleton, nearest_skeletons

            hits = nearest_skeletons(
                self.skeleton_bank, query_prompt or prompt, k=k
            )
            if hits:
                skeleton = format_retrieved_skeleton(hits[0].openui)
        dm = design_md if self.config.design_md_in_context else None
        return format_context_text(
            prompt,
            dm,
            budget=self.config.design_md_budget,
            schema=schema,
            retrieved_skeleton=skeleton,
        )
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

    def _ltr_canvases(self, length: int) -> list[int]:
        stages = [s for s in self.config.grammar_ltr_stages if 1 < s <= length]
        if not stages or stages[-1] != length:
            stages = [*stages, length] if stages else [length]
        seen: set[int] = set()
        canvases: list[int] = []
        for s in stages:
            if s not in seen:
                seen.add(s)
                canvases.append(s)
        return canvases

    def _greedy_ltr_decode(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Left-to-right argmax decode (batch size 1 wrapper)."""
        return self._greedy_ltr_decode_batch(ctx, ctx_pad, length)

    def _greedy_ltr_decode_batch(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """
        Batched LTR argmax decode with progressive canvases.

        Finished sequences (EOS) are padded and skipped; remaining rows share
        transformer forwards — large win for eval throughput.
        """
        bsz = int(ctx.size(0))
        device = self.device_name
        tok = self.tokenizer
        bias = float(self.config.structural_bias or 0.0)
        canvases = self._ltr_canvases(length)

        ids: torch.Tensor | None = None
        active = torch.ones(bsz, dtype=torch.bool, device=device)
        start_t = 1

        for canvas in canvases:
            if ids is None:
                ids = torch.full(
                    (bsz, canvas),
                    tok.mask_id,
                    dtype=torch.long,
                    device=device,
                )
                ids[:, 0] = tok.bos_id
            else:
                extra = canvas - ids.size(1)
                if extra > 0:
                    pad = torch.full(
                        (bsz, extra),
                        tok.mask_id,
                        dtype=torch.long,
                        device=device,
                    )
                    # Finished sequences should stay padded, not re-masked.
                    if (~active).any():
                        pad = pad.clone()
                        pad[~active] = tok.pad_id
                    ids = torch.cat([ids, pad], dim=1)

            for t in range(start_t, canvas):
                if not bool(active.any()):
                    break
                active_idx = active.nonzero(as_tuple=False).flatten()
                if active_idx.numel() == bsz:
                    logits = self.denoiser(
                        ids, ctx, pad_id=tok.pad_id, ctx_pad_mask=ctx_pad
                    )
                    if bias:
                        logits = apply_structural_bias(logits, tok, bias=bias)
                    row = logits[:, t, :].clone()
                else:
                    sub_ids = ids.index_select(0, active_idx)
                    sub_ctx = ctx.index_select(0, active_idx)
                    sub_pad = ctx_pad.index_select(0, active_idx)
                    logits = self.denoiser(
                        sub_ids, sub_ctx, pad_id=tok.pad_id, ctx_pad_mask=sub_pad
                    )
                    if bias:
                        logits = apply_structural_bias(logits, tok, bias=bias)
                    row = torch.full(
                        (bsz, logits.size(-1)),
                        -1e9,
                        device=device,
                        dtype=logits.dtype,
                    )
                    row.index_copy_(0, active_idx, logits[:, t, :])
                row = row.clone()
                row[:, tok.mask_id] = -1e9
                row[:, tok.pad_id] = -1e9
                pred = row.argmax(dim=-1)
                ids[:, t] = torch.where(active, pred, ids[:, t])
                hit_eos = active & pred.eq(tok.eos_id)
                if bool(hit_eos.any()) and t + 1 < canvas:
                    for b in hit_eos.nonzero(as_tuple=False).flatten().tolist():
                        ids[b, t + 1 :] = tok.pad_id
                active = active & ~pred.eq(tok.eos_id)
            start_t = canvas
            if not bool(active.any()):
                break

        assert ids is not None
        return ids

    def _context_prompts(
        self,
        prompts: list[str],
        golds: list[ExampleRecord | None] | None = None,
        design_mds: list[str | None] | None = None,
    ) -> list[str]:
        out: list[str] = []
        for i, prompt in enumerate(prompts):
            dm = design_mds[i] if design_mds else None
            if dm is None and golds and golds[i] is not None:
                dm = golds[i].design_md  # type: ignore[union-attr]
            out.append(self._format_one_context(prompt, dm, query_prompt=prompt))
        return out

    def _repair_ltr_texts(
        self,
        texts: list[str],
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
    ) -> list[str]:
        """Re-decode rows that fail stream_check using constrained LTR."""
        from concurrent.futures import ThreadPoolExecutor

        def _check(text: str) -> tuple[bool, str]:
            try:
                status = stream_check(text)
                if status.serialized and status.complete_ok:
                    ser = status.serialized.strip()
                    compact = ser.replace(" ", "")
                    if "Stack([])" not in compact and "Card([])" not in compact:
                        return True, ser
            except Exception:  # noqa: BLE001
                pass
            return False, text

        # Parallel Node stream_check — grammar bridge is process-bound, so threads
        # overlap Python wait when the Node CLI is the bottleneck.
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(texts)))) as pool:
            checked = list(pool.map(_check, texts))

        repaired: list[str] = []
        for i, (ok, text) in enumerate(checked):
            if ok:
                repaired.append(text)
                continue
            row_ids = torch.full(
                (1, length),
                self.tokenizer.mask_id,
                dtype=torch.long,
                device=self.device_name,
            )
            row_ids[0, 0] = self.tokenizer.bos_id
            unknown = row_ids.eq(self.tokenizer.mask_id)
            filled = self._constrained_ltr_repair(
                row_ids,
                unknown,
                ctx[i : i + 1],
                ctx_pad[i : i + 1],
            )
            repaired.append(self._decode_ids(filled[0]))
        return repaired

    def _pick_best_of_n(
        self,
        candidates: list[str],
        gold: ExampleRecord | None,
    ) -> str:
        if len(candidates) == 1:
            return candidates[0]
        from slm_training.preference import composite_reward

        best = candidates[0]
        best_score = -1.0
        for cand in candidates:
            score = float(composite_reward(cand, gold=gold, design_md=None))
            if score > best_score:
                best_score = score
                best = cand
        return best

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[str],
        golds: list[ExampleRecord | None] | None = None,
        *,
        max_len: int | None = None,
        grammar_constrained: bool | None = None,
        design_mds: list[str | None] | None = None,
    ) -> list[str]:
        """Batched generate — preferred for eval throughput."""
        self.eval()
        if not prompts:
            return []
        n_samples = max(1, int(getattr(self.config, "best_of_n", 1) or 1))
        if n_samples > 1:
            pools: list[list[str]] = [[] for _ in prompts]
            prev = self.config.best_of_n
            self.config.best_of_n = 1
            try:
                for _ in range(n_samples):
                    sample = self._generate_batch_once(
                        prompts,
                        golds,
                        max_len=max_len,
                        grammar_constrained=grammar_constrained,
                        design_mds=design_mds,
                    )
                    for i, text in enumerate(sample):
                        pools[i].append(text)
            finally:
                self.config.best_of_n = prev
            out: list[str] = []
            for i, cands in enumerate(pools):
                gold = golds[i] if golds else None
                out.append(self._pick_best_of_n(cands, gold))
            return out
        return self._generate_batch_once(
            prompts,
            golds,
            max_len=max_len,
            grammar_constrained=grammar_constrained,
            design_mds=design_mds,
        )

    def _generate_batch_once(
        self,
        prompts: list[str],
        golds: list[ExampleRecord | None] | None = None,
        *,
        max_len: int | None = None,
        grammar_constrained: bool | None = None,
        design_mds: list[str | None] | None = None,
    ) -> list[str]:
        use_grammar = (
            self.config.grammar_constrained
            if grammar_constrained is None
            else grammar_constrained
        )
        length = max_len or self.gen_len or self.config.max_target_len
        length = max(8, min(int(length), self.config.max_target_len))
        ctx_prompts = self._context_prompts(prompts, golds=golds, design_mds=design_mds)
        ctx, ctx_pad = self._encode_context(ctx_prompts)

        if use_grammar and self.config.grammar_ltr_primary:
            repair_len = min(length, max(8, int(self.config.grammar_ltr_max_tokens)))
            ids = self._greedy_ltr_decode_batch(ctx, ctx_pad, repair_len)
            texts = [self._decode_ids(ids[i]) for i in range(ids.size(0))]
            if self.config.grammar_ltr_repair:
                texts = self._repair_ltr_texts(texts, ctx, ctx_pad, repair_len)
            if self.config.grammar_finalize_validate:
                finalized: list[str] = []
                for text in texts:
                    try:
                        from slm_training.dsl.parser import validate

                        program = validate(text)
                        ser = (program.serialized or text).strip()
                        compact = ser.replace(" ", "")
                        if "Stack([])" not in compact and "Card([])" not in compact:
                            finalized.append(ser)
                            continue
                    except Exception:  # noqa: BLE001
                        pass
                    finalized.append(text)
                return finalized
            return texts

        # Fall back to per-item MaskGIT for non-LTR-primary path.
        out: list[str] = []
        for i, prompt in enumerate(prompts):
            gold = golds[i] if golds else None
            dm = design_mds[i] if design_mds else None
            out.append(
                self.generate(
                    prompt,
                    gold=gold,
                    max_len=length,
                    grammar_constrained=use_grammar,
                    design_md=dm,
                )
            )
        return out

    def _generate_maskgit_one(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        *,
        use_grammar: bool,
    ) -> str:
        """Single-sequence MaskGIT unmasking (+ optional grammar repair)."""
        device = self.device_name
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
            mode = str(getattr(self.config, "parallel_unmask", "adaptive") or "topk")
            flat_idx = select_unmask_indices(
                conf,
                unknown,
                step=step,
                steps=steps,
                mode=mode,
            )
            newly: list[int] = []
            for idx in flat_idx:
                b = idx // length
                t = idx % length
                if unknown[b, t]:
                    ids[b, t] = pred[b, t]
                    unknown[b, t] = False
                    if b == 0:
                        newly.append(t)
            _ = remaining  # kept for readability / future logging

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
                    ser = status.serialized.strip()
                    if "Stack([])" not in ser.replace(" ", ""):
                        return ser
            except Exception:  # noqa: BLE001
                pass
            if not self.config.grammar_ltr_repair:
                return text
            try:
                from slm_training.dsl.parser import validate

                program = validate(text)
                ser = (program.serialized or text).strip()
                if "Stack([])" not in ser.replace(" ", ""):
                    return ser
            except Exception:  # noqa: BLE001
                pass
            repair_len = min(length, max(8, int(self.config.grammar_ltr_max_tokens)))
            repaired = torch.full(
                (1, repair_len),
                self.tokenizer.mask_id,
                dtype=torch.long,
                device=device,
            )
            repaired[0, 0] = self.tokenizer.bos_id
            unknown_r = repaired.eq(self.tokenizer.mask_id)
            repaired = self._constrained_ltr_repair(
                repaired, unknown_r, ctx, ctx_pad
            )
            text2 = self._decode_ids(repaired[0])
            try:
                status = stream_check(text2)
                if status.serialized and status.complete_ok:
                    ser = status.serialized.strip()
                    if "Stack([])" not in ser.replace(" ", ""):
                        return ser
            except Exception:  # noqa: BLE001
                pass
            return text2
        return text

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        gold: ExampleRecord | None = None,
        max_len: int | None = None,
        grammar_constrained: bool | None = None,
        design_md: str | None = None,
    ) -> str:
        """Generate OpenUI for one prompt (batched LTR when enabled)."""
        return self.generate_batch(
            [prompt],
            golds=[gold],
            max_len=max_len,
            grammar_constrained=grammar_constrained,
            design_mds=[design_md],
        )[0]

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
        if isinstance(raw_cfg.get("grammar_ltr_stages"), list):
            raw_cfg["grammar_ltr_stages"] = tuple(raw_cfg["grammar_ltr_stages"])
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
