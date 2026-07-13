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
    force_emit_token_id,
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
    slot_contract: list[str] | None = None,
) -> str:
    """Concatenate prompt with optional schema / skeleton / slot contract / DESIGN.md."""
    prompt = (prompt or "").strip()
    parts = [prompt] if prompt else []
    if schema and schema.strip():
        parts.append(f"---SCHEMA---\n{schema.strip()[: min(600, budget)]}")
    if retrieved_skeleton and retrieved_skeleton.strip():
        parts.append(
            f"---RETRIEVED_SKELETON---\n{retrieved_skeleton.strip()[: min(400, budget)]}"
        )
    if slot_contract:
        slots = ", ".join(slot_contract)
        parts.append(f"---SLOT_CONTRACT---\n{slots[: min(800, budget)]}")
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
    slot_contract_in_context: bool = False
    slot_contract_constrained_decode: bool = False
    retrieval_k: int = 0
    best_of_n: int = 1
    seed: int = 0
    # Accelerator / SOTA decode knobs
    use_compile: bool = False
    compile_mode: str = "default"
    use_amp: bool = False
    # MaskGIT parallel unmask: topk | confidence | adaptive (mean-field-lite)
    parallel_unmask: str = "adaptive"
    # Train-speed: cache frozen HF backbone hiddens + formatted context strings.
    cache_context: bool = True
    # Fuse LTR suffix masks into the MaskGIT canvas (one denoiser forward).
    fuse_ltr_loss: bool = True
    # Grammar fast-path (decode); aux weight applied in training_loss when >0.
    grammar_fastpath: bool = True
    grammar_fastpath_mode: str = "hybrid"  # force | mask | hybrid
    grammar_draft_window: int = 8
    fastpath_aux_weight: float = 0.0
    fastpath_gate_threshold: float = 0.5


def _pad_batch(seqs: list[list[int]], pad_id: int, device: str | torch.device | None = None) -> torch.Tensor:
    max_len = max((len(s) for s in seqs), default=1)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            out[i, : len(s)] = torch.as_tensor(s, dtype=torch.long)
    if device is not None:
        out = out.to(device)
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
        # Train-time caches (formatted context string keyed by record id).
        self._context_text_cache: dict[str, str] = {}
        self._target_ids_cache: dict[str, list[int]] = {}
        self._placeholder_token_ids: set[int] | None = None
        self._slot_contracts: list[list[str] | None] | None = None
        self.to(device)

    def clear_train_caches(self) -> None:
        self._context_text_cache.clear()
        self._target_ids_cache.clear()
        if is_hf_context(self.context) and hasattr(self.context, "clear_backbone_cache"):
            self.context.clear_backbone_cache()  # type: ignore[union-attr]

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def _encode_context(
        self,
        prompts: list[str],
        *,
        cache_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from slm_training.telemetry import timed

        with timed("context_encode"):
            if is_hf_context(self.context):
                assert isinstance(self.context, HFContextEncoder)
                self.context.cache_backbone = bool(
                    getattr(self.config, "cache_context", True)
                )
                return self.context.forward_prompts(
                    prompts,
                    max_len=self.config.max_prompt_len,
                    device=self.device_name,
                    cache_keys=cache_keys if self.config.cache_context else None,
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
        """Return noisy_ids and boolean mask of positions to predict (vectorized)."""
        bsz, seq = target_ids.shape
        device = target_ids.device
        frozen = target_ids.eq(self.tokenizer.pad_id) | target_ids.eq(self.tokenizer.bos_id)
        # Sample a per-row mask rate, then Bernoulli over valid positions.
        rates = torch.empty(bsz, 1, device=device).uniform_(
            self.config.mask_min, self.config.mask_max
        )
        rand = torch.rand(bsz, seq, device=device)
        noise = (rand < rates) & (~frozen)
        # Ensure at least one predictable token per non-empty row.
        for i in range(bsz):
            if frozen[i].all():
                continue
            if not bool(noise[i].any()):
                valid = (~frozen[i]).nonzero(as_tuple=False).view(-1)
                if valid.numel():
                    noise[i, int(valid[self._rng.randrange(valid.numel())])] = True
        noisy = target_ids.clone()
        noisy[noise] = self.tokenizer.mask_id
        return noisy, noise

    def _merge_ltr_suffix_mask(
        self, target_ids: torch.Tensor, noisy: torch.Tensor, predict_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Force-mask a random suffix; return ltr-only positions for loss weighting."""
        bsz, seq = target_ids.shape
        ltr_suffix = torch.zeros_like(predict_mask)
        for i in range(bsz):
            cut = self._rng.randint(1, max(1, seq - 1))
            for j in range(cut, seq):
                if int(target_ids[i, j]) == self.tokenizer.pad_id:
                    break
                ltr_suffix[i, j] = True
                if not bool(predict_mask[i, j].item()):
                    predict_mask[i, j] = True
                    noisy[i, j] = self.tokenizer.mask_id
        return noisy, predict_mask, ltr_suffix

    def _placeholder_ids(self) -> set[int]:
        if self._placeholder_token_ids is None:
            ids: set[int] = set()
            for tok, tid in self.tokenizer.token_to_id.items():
                if tok in {":", "."}:
                    ids.add(tid)
                elif tok and tok[0].islower() and tok.isidentifier():
                    ids.add(tid)
                elif tok.startswith('":') or (
                    tok.startswith('"') and ":" in tok
                ):
                    ids.add(tid)
            self._placeholder_token_ids = ids
        return self._placeholder_token_ids

    def forward(self, batch: list[ExampleRecord]) -> float:
        self.train()
        loss = self.training_loss(batch)
        return float(loss.detach().cpu())

    def training_loss(self, batch: list[ExampleRecord]) -> torch.Tensor:
        self.train()
        cache_on = bool(getattr(self.config, "cache_context", True))
        prompts: list[str] = []
        cache_keys: list[str] = []
        targets: list[list[int]] = []
        for r in batch:
            key = r.id or r.prompt
            cache_keys.append(key)
            if cache_on and key in self._context_text_cache:
                prompts.append(self._context_text_cache[key])
            else:
                text = self._format_one_context(
                    r.prompt,
                    r.design_md,
                    query_prompt=r.prompt,
                    slot_contract=list(r.placeholders or [])
                    if getattr(self.config, "slot_contract_in_context", False)
                    else None,
                )
                if cache_on:
                    self._context_text_cache[key] = text
                prompts.append(text)
            if cache_on and key in self._target_ids_cache:
                targets.append(self._target_ids_cache[key])
            else:
                ids = self.tokenizer.encode(r.openui)[: self.config.max_target_len]
                if cache_on:
                    self._target_ids_cache[key] = ids
                targets.append(ids)

        target_ids = _pad_batch(
            targets, self.tokenizer.pad_id, device=self.device_name
        )
        ctx, ctx_pad = self._encode_context(prompts, cache_keys=cache_keys)
        noisy, predict_mask = self._mask_targets(target_ids)

        ltr_w = float(self.config.ltr_loss_weight or 0.0)
        fuse = bool(getattr(self.config, "fuse_ltr_loss", True))
        ltr_suffix = torch.zeros_like(predict_mask)
        if ltr_w > 0.0 and target_ids.size(1) >= 2 and fuse:
            noisy, predict_mask, ltr_suffix = self._merge_ltr_suffix_mask(
                target_ids, noisy, predict_mask
            )

        from slm_training.telemetry import timed

        with timed("denoiser_forward"):
            logits = self.denoiser(
                noisy, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
        if predict_mask.any():
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = target_ids.reshape(-1)
            ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            weights = torch.ones_like(ce)
            if ltr_w > 0.0 and fuse and ltr_suffix.any():
                suffix_flat = ltr_suffix.reshape(-1)
                weights = weights + (ltr_w * suffix_flat.float())
            mask_flat = predict_mask.reshape(-1)
            mask_loss = (ce * weights)[mask_flat].mean()
        else:
            mask_loss = logits.sum() * 0.0

        fid_w = float(getattr(self.config, "fidelity_loss_weight", 0.0) or 0.0)
        if fid_w > 0.0 and predict_mask.any():
            ph_ids: set[int] = set()
            for r in batch:
                for ph in r.placeholders or []:
                    for tid in self.tokenizer.encode(f'"{ph}"', add_special=False):
                        ph_ids.add(tid)
            if not ph_ids:
                ph_ids = self._placeholder_ids()
            if ph_ids:
                ph_mask = predict_mask.clone()
                # Vectorized membership via isin.
                ph_tensor = torch.tensor(
                    sorted(ph_ids), device=target_ids.device, dtype=target_ids.dtype
                )
                ph_mask &= torch.isin(target_ids, ph_tensor)
                if ph_mask.any():
                    fid_loss = F.cross_entropy(logits[ph_mask], target_ids[ph_mask])
                    mask_loss = mask_loss + fid_w * fid_loss

        # Legacy second-forward LTR when fuse disabled.
        if ltr_w > 0.0 and target_ids.size(1) >= 2 and not fuse:
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
            mask_loss = mask_loss + ltr_w * ltr_loss

        aux_w = float(getattr(self.config, "fastpath_aux_weight", 0.0) or 0.0)
        if aux_w > 0.0 and getattr(self.config, "grammar_fastpath", False):
            try:
                from slm_training.grammar_fastpath.losses import force_align_loss

                aux = force_align_loss(
                    logits, target_ids, self.tokenizer, pad_id=self.tokenizer.pad_id
                )
                mask_loss = mask_loss + aux_w * aux
            except Exception:  # noqa: BLE001
                pass

        return mask_loss

    def _format_one_context(
        self,
        prompt: str,
        design_md: str | None,
        *,
        query_prompt: str | None = None,
        slot_contract: list[str] | None = None,
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
        contract = (
            slot_contract
            if getattr(self.config, "slot_contract_in_context", False)
            else None
        )
        return format_context_text(
            prompt,
            dm,
            budget=self.config.design_md_budget,
            schema=schema,
            retrieved_skeleton=skeleton,
            slot_contract=contract,
        )
    def _decode_ids(self, ids_1d: torch.Tensor) -> str:
        token_ids = ids_1d.tolist()
        if self.tokenizer.eos_id in token_ids[1:]:
            end = token_ids.index(self.tokenizer.eos_id, 1)
            token_ids = token_ids[: end + 1]
        return self.tokenizer.decode(token_ids).strip()

    @staticmethod
    def _canonical_valid_openui(text: str) -> str | None:
        """Return serialized OpenUI if parseable and non-trivial; else None."""
        try:
            from slm_training.dsl.parser import validate
        except Exception:  # noqa: BLE001
            return None
        try:
            program = validate(text)
        except Exception:  # noqa: BLE001
            return None
        ser = (program.serialized or text).strip()
        compact = ser.replace(" ", "")
        if "Stack([])" in compact or "Stack([]," in compact:
            return None
        if "Card([])" in compact:
            return None
        if "root=" not in compact and "root =" not in ser:
            return None
        return ser

    def _minimal_valid_openui(self) -> str | None:
        """Deterministic vocab-backed valid program when model decode cannot certify."""
        candidates = [
            'root = Button(":cta.label")\n',
            'root = TextContent(":hero.title")\n',
            'root = Stack([cta])\ncta = Button(":cta.label")\n',
            'root = Card([title])\ntitle = TextContent(":hero.title")\n',
        ]
        for raw in candidates:
            ser = self._canonical_valid_openui(raw)
            if ser is None:
                continue
            ids = self.tokenizer.encode(ser, add_special=False)
            if not ids or self.tokenizer.unk_id in ids:
                continue
            return ser
        return None

    def _ltr_repair_from_bos(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
    ) -> str:
        """Speculative LTR decode from BOS with force-emit + constrained picks."""
        device = self.device_name
        repair_len = min(length, max(8, int(self.config.grammar_ltr_max_tokens)))
        repaired = torch.full(
            (1, repair_len),
            self.tokenizer.mask_id,
            dtype=torch.long,
            device=device,
        )
        repaired[0, 0] = self.tokenizer.bos_id
        unknown_r = repaired.eq(self.tokenizer.mask_id)
        repaired = self._constrained_ltr_repair(repaired, unknown_r, ctx, ctx_pad)
        return self._decode_ids(repaired[0])

    def _ensure_valid_openui(
        self,
        text: str,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        *,
        attempts: int = 3,
    ) -> str:
        """
        Repair toward a valid OpenUI string when grammar-constrained.

        Uses DFA-constrained LTR repair (pseudo speculative decoding). When
        ``grammar_finalize_validate`` is set (playground / hard certify), never
        returns invalid OpenUI — falls back to a minimal certified program or
        raises. When finalize is off (default eval), returns the best repaired
        text so parse_rate reflects real decode quality.
        """
        ser = self._canonical_valid_openui(text)
        if ser is not None:
            return ser
        last = text
        for _ in range(max(1, attempts)):
            last = self._ltr_repair_from_bos(ctx, ctx_pad, length)
            ser = self._canonical_valid_openui(last)
            if ser is not None:
                return ser
        if self.config.grammar_finalize_validate:
            fallback = self._minimal_valid_openui()
            if fallback is not None:
                return fallback
            raise RuntimeError(
                "grammar_finalize_validate: model could not produce a complete valid OpenUI program"
            )
        return last

    def _constrained_ltr_repair(
        self,
        ids: torch.Tensor,
        unknown: torch.Tensor,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        *,
        slot_contract: list[str] | None = None,
    ) -> torch.Tensor:
        """Fill remaining masks left-to-right with streaming-parser filtering."""
        use_contract = bool(
            getattr(self.config, "slot_contract_constrained_decode", False)
        )
        contract = slot_contract if use_contract else None
        length = ids.size(1)
        use_fast = bool(getattr(self.config, "grammar_fastpath", True))
        for t in range(length):
            if not bool(unknown[0, t].item()):
                continue
            prefix = ids[0, :t].tolist()
            forced = (
                force_emit_token_id(self.tokenizer, prefix)
                if use_fast
                else None
            )
            if forced is None:
                logits = self.denoiser(
                    ids, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
                )
                if self.config.structural_bias:
                    logits = apply_structural_bias(
                        logits,
                        self.tokenizer,
                        bias=self.config.structural_bias,
                    )
                choice = pick_constrained_token(
                    logits[0, t],
                    self.tokenizer,
                    prefix,
                    top_k=self.config.grammar_top_k,
                    slot_contract=contract,
                )
            else:
                # Zero logits stand-in; forced id short-circuits inside picker.
                choice = pick_constrained_token(
                    torch.zeros(
                        self.tokenizer.vocab_size,
                        device=ids.device,
                    ),
                    self.tokenizer,
                    prefix,
                    top_k=self.config.grammar_top_k,
                    forced_token_id=forced,
                    slot_contract=contract,
                )
            if choice is None:
                # No legal continuation — pad out and stop rather than emit garbage.
                ids[0, t:] = self.tokenizer.pad_id
                unknown[0, t:] = False
                break
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

            use_fast = bool(getattr(self.config, "grammar_fastpath", True))
            for t in range(start_t, canvas):
                if not bool(active.any()):
                    break
                active_idx = active.nonzero(as_tuple=False).flatten()
                forced_map: dict[int, int] = {}
                need_model: list[int] = []
                if use_fast:
                    for bi in active_idx.tolist():
                        forced = force_emit_token_id(
                            tok, ids[bi, :t].tolist()
                        )
                        if forced is not None:
                            forced_map[bi] = forced
                        else:
                            need_model.append(bi)
                else:
                    need_model = active_idx.tolist()

                pred = ids[:, t].clone()
                for bi, fid in forced_map.items():
                    contract = (
                        self._slot_contracts[bi]
                        if self._slot_contracts and bi < len(self._slot_contracts)
                        else None
                    )
                    if not getattr(self.config, "slot_contract_constrained_decode", False):
                        contract = None
                    choice = pick_constrained_token(
                        torch.zeros(tok.vocab_size, device=device),
                        tok,
                        ids[bi, :t].tolist(),
                        top_k=self.config.grammar_top_k,
                        forced_token_id=fid,
                        slot_contract=contract,
                    )
                    if choice is None:
                        need_model.append(bi)
                    else:
                        pred[bi] = choice

                if need_model:
                    need_t = torch.tensor(
                        need_model, device=device, dtype=torch.long
                    )
                    if need_t.numel() == bsz:
                        logits = self.denoiser(
                            ids, ctx, pad_id=tok.pad_id, ctx_pad_mask=ctx_pad
                        )
                        if bias:
                            logits = apply_structural_bias(
                                logits, tok, bias=bias
                            )
                        row = logits[:, t, :].clone()
                    else:
                        sub_ids = ids.index_select(0, need_t)
                        sub_ctx = ctx.index_select(0, need_t)
                        sub_pad = ctx_pad.index_select(0, need_t)
                        logits = self.denoiser(
                            sub_ids,
                            sub_ctx,
                            pad_id=tok.pad_id,
                            ctx_pad_mask=sub_pad,
                        )
                        if bias:
                            logits = apply_structural_bias(
                                logits, tok, bias=bias
                            )
                        row = torch.full(
                            (bsz, logits.size(-1)),
                            -1e9,
                            device=device,
                            dtype=logits.dtype,
                        )
                        row.index_copy_(0, need_t, logits[:, t, :])
                    row = row.clone()
                    row[:, tok.mask_id] = -1e9
                    row[:, tok.pad_id] = -1e9
                    for bi in need_model:
                        contract = (
                            self._slot_contracts[bi]
                            if self._slot_contracts and bi < len(self._slot_contracts)
                            else None
                        )
                        if not getattr(
                            self.config, "slot_contract_constrained_decode", False
                        ):
                            contract = None
                        choice = pick_constrained_token(
                            row[bi],
                            tok,
                            ids[bi, :t].tolist(),
                            top_k=self.config.grammar_top_k,
                            slot_contract=contract,
                        )
                        if choice is None:
                            # No legal token — end sequence rather than emit garbage.
                            pred[bi] = tok.eos_id
                        else:
                            pred[bi] = choice

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
        use_contract = bool(getattr(self.config, "slot_contract_in_context", False))
        for i, prompt in enumerate(prompts):
            dm = design_mds[i] if design_mds else None
            contract: list[str] | None = None
            if golds and golds[i] is not None:
                gold = golds[i]
                if dm is None:
                    dm = gold.design_md  # type: ignore[union-attr]
                if use_contract:
                    contract = list(gold.placeholders or [])  # type: ignore[union-attr]
            out.append(
                self._format_one_context(
                    prompt,
                    dm,
                    query_prompt=prompt,
                    slot_contract=contract,
                )
            )
        return out

    def _repair_ltr_texts(
        self,
        texts: list[str],
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        *,
        slot_contracts: list[list[str] | None] | None = None,
    ) -> list[str]:
        """Re-decode rows that fail stream_check using constrained LTR."""
        from concurrent.futures import ThreadPoolExecutor

        def _check(text: str, contract: list[str] | None) -> tuple[bool, str]:
            try:
                status = stream_check(text)
                if status.serialized and status.complete_ok:
                    ser = status.serialized.strip()
                    compact = ser.replace(" ", "")
                    if "Stack([])" not in compact and "Card([])" not in compact:
                        if contract:
                            from slm_training.dsl.placeholders import extract_placeholders

                            preds = set(extract_placeholders(ser))
                            allowed = {p if p.startswith(":") else f":{p}" for p in contract}
                            if preds and not preds.issubset(allowed):
                                return False, text
                        return True, ser
            except Exception:  # noqa: BLE001
                pass
            return False, text

        # Parallel Node stream_check — grammar bridge is process-bound, so threads
        # overlap Python wait when the Node CLI is the bottleneck.
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(texts)))) as pool:
            checked = list(
                pool.map(
                    lambda item: _check(item[0], item[1]),
                    [
                        (
                            text,
                            slot_contracts[i]
                            if slot_contracts and i < len(slot_contracts)
                            else None,
                        )
                        for i, text in enumerate(texts)
                    ],
                )
            )

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
                slot_contract=(
                    slot_contracts[i]
                    if slot_contracts and i < len(slot_contracts)
                    else None
                ),
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
        from slm_training.telemetry import timed

        self.eval()
        if not prompts:
            return []
        n_samples = max(1, int(getattr(self.config, "best_of_n", 1) or 1))
        if n_samples > 1:
            pools: list[list[str]] = [[] for _ in prompts]
            prev = self.config.best_of_n
            self.config.best_of_n = 1
            try:
                with timed("generate_batch"):
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
            with timed("best_of_n_rank"):
                for i, cands in enumerate(pools):
                    gold = golds[i] if golds else None
                    out.append(self._pick_best_of_n(cands, gold))
            return out
        with timed("generate_batch"):
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
        use_contract_decode = bool(
            getattr(self.config, "slot_contract_constrained_decode", False)
        )
        if use_contract_decode and golds:
            self._slot_contracts = [
                list(g.placeholders or []) if g else None for g in golds
            ]
        else:
            self._slot_contracts = None
        ctx_prompts = self._context_prompts(prompts, golds=golds, design_mds=design_mds)
        ctx, ctx_pad = self._encode_context(ctx_prompts)

        if use_grammar and self.config.grammar_ltr_primary:
            repair_len = min(length, max(8, int(self.config.grammar_ltr_max_tokens)))
            ids = self._greedy_ltr_decode_batch(ctx, ctx_pad, repair_len)
            texts = [self._decode_ids(ids[i]) for i in range(ids.size(0))]
            if self.config.grammar_ltr_repair:
                texts = self._repair_ltr_texts(
                    texts,
                    ctx,
                    ctx_pad,
                    repair_len,
                    slot_contracts=self._slot_contracts,
                )
            # Certify when grammar-constrained (finalize controls canned fallback).
            certified: list[str] = []
            for i, text in enumerate(texts):
                certified.append(
                    self._ensure_valid_openui(
                        text, ctx[i : i + 1], ctx_pad[i : i + 1], length
                    )
                )
            return certified

        # Fall back to per-item MaskGIT for non-LTR-primary path.
        out: list[str] = []
        for i in range(len(prompts)):
            contract = (
                self._slot_contracts[i]
                if self._slot_contracts and i < len(self._slot_contracts)
                else None
            )
            out.append(
                self._generate_maskgit_one(
                    ctx[i : i + 1],
                    ctx_pad[i : i + 1],
                    length,
                    use_grammar=use_grammar,
                    slot_contract=contract,
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
        slot_contract: list[str] | None = None,
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
            use_fast = bool(getattr(self.config, "grammar_fastpath", True))
            mode = str(
                getattr(self.config, "grammar_fastpath_mode", "hybrid") or "hybrid"
            )
            admit_on = use_grammar and use_fast and mode in {"mask", "hybrid"}
            engine = None
            if admit_on:
                try:
                    from slm_training.grammar_fastpath import admit_fill, engine_for_dsl
                    from slm_training.models.grammar import active_dsl

                    engine = engine_for_dsl(active_dsl())
                    if engine is None:
                        admit_on = False
                except Exception:  # noqa: BLE001
                    engine = None
                    admit_on = False

            for idx in flat_idx:
                b = idx // length
                t = idx % length
                if not unknown[b, t]:
                    continue
                prefix = ids[b, :t].tolist()
                forced = (
                    force_emit_token_id(self.tokenizer, prefix)
                    if use_fast
                    else None
                )
                if forced is not None or use_grammar:
                    # Speculative / constrained pick — never commit illegal tokens.
                    logits_t = logits[b, t]
                    choice = pick_constrained_token(
                        logits_t,
                        self.tokenizer,
                        prefix,
                        top_k=self.config.grammar_top_k,
                        forced_token_id=forced,
                        slot_contract=slot_contract
                        if getattr(self.config, "slot_contract_constrained_decode", False)
                        else None,
                    )
                    if choice is None:
                        continue  # leave masked for LTR repair
                    candidate = choice
                else:
                    candidate = int(pred[b, t].item())
                if admit_on and engine is not None and b == 0:
                    trial = ids[0].tolist()
                    trial[t] = candidate
                    try:
                        if not admit_fill(engine, self.tokenizer, trial):
                            continue  # leave masked; try later / repair
                    except Exception:  # noqa: BLE001
                        continue  # reject on admit probe failure
                ids[b, t] = candidate
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
                ids = self._constrained_ltr_repair(
                    ids,
                    unknown,
                    ctx,
                    ctx_pad,
                    slot_contract=slot_contract,
                )
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
            return self._ensure_valid_openui(text, ctx, ctx_pad, length)
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
