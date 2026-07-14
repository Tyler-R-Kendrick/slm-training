"""Lightweight ONNX inference adapter for the deployed playground."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime as ort

from slm_training.dsl.parser import validate
from slm_training.models.grammar import (
    dfa_admits_token,
    force_emit_token_id,
    structural_token_ids,
)
from slm_training.models.tokenizer import OpenUITokenizer


class OnnxTwoTowerModel:
    """Inference-only TwoTower model with the interface used by PlaygroundService."""

    def __init__(
        self,
        *,
        tokenizer: OpenUITokenizer,
        config: SimpleNamespace,
        gen_len: int,
        context_path: Path,
        denoiser_path: Path,
    ) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.context_session = ort.InferenceSession(
            str(context_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.denoiser_session = ort.InferenceSession(
            str(denoiser_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = tokenizer
        self.config = config
        self.gen_len = gen_len

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path | str, device: str = "cpu"
    ) -> OnnxTwoTowerModel:
        if device != "cpu":
            raise ValueError("ONNX playground inference supports device='cpu' only")
        checkpoint = Path(checkpoint)
        meta = json.loads(checkpoint.with_suffix(".meta.json").read_text())
        raw_config = dict(meta.get("config") or {})
        defaults: dict[str, Any] = {
            "grammar_constrained": True,
            "grammar_top_k": 16,
            "structural_bias": 1.25,
            "grammar_ltr_max_tokens": 64,
            "grammar_ltr_primary": True,
            "grammar_ltr_repair": True,
            "grammar_finalize_validate": True,
            "grammar_fastpath": True,
            "grammar_prefer_structural": True,
            "grammar_sample_decode": False,
            "grammar_sample_temperature": 0.8,
            "max_prompt_len": 256,
            "max_target_len": 256,
            "design_md_budget": 1800,
        }
        defaults.update(raw_config)
        stem = checkpoint.with_suffix("")
        context_path = stem.with_suffix(".context.onnx")
        denoiser_path = stem.with_suffix(".denoiser.onnx")
        for artifact in (context_path, denoiser_path):
            if not artifact.is_file():
                raise FileNotFoundError(f"missing ONNX inference artifact: {artifact}")
        return cls(
            tokenizer=OpenUITokenizer.load(checkpoint.with_suffix(".tokenizer.json")),
            config=SimpleNamespace(**defaults),
            gen_len=int(meta.get("gen_len") or defaults["max_target_len"]),
            context_path=context_path,
            denoiser_path=denoiser_path,
        )

    def eval(self) -> OnnxTwoTowerModel:
        return self

    def _format_context(self, prompt: str, design_md: str | None) -> str:
        prompt = prompt.strip()
        if not design_md or not design_md.strip():
            return prompt
        budget = int(getattr(self.config, "design_md_budget", 1800))
        return f"{prompt}\n\n---DESIGN.md---\n{design_md.strip()[:budget]}"

    def _encode_context(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        ids = self.tokenizer.encode(text)[: int(self.config.max_prompt_len)]
        if not ids:
            ids = [self.tokenizer.bos_id]
        input_ids = np.asarray([ids], dtype=np.int64)
        pad_mask = input_ids == self.tokenizer.pad_id
        context = self.context_session.run(
            ["context"], {"input_ids": input_ids}
        )[0]
        return context, pad_mask

    def _certify(self, text: str) -> str | None:
        try:
            program = validate(text)
        except Exception:  # noqa: BLE001
            return None
        serialized = (program.serialized or text).strip()
        compact = serialized.replace(" ", "")
        if (
            "root=" not in compact
            or "Stack([]" in compact
            or "Card([]" in compact
        ):
            return None
        return serialized

    def _pick_constrained_token(
        self,
        logits: np.ndarray,
        prefix: list[int],
        forced_token_id: int | None,
    ) -> int | None:
        blocked = {
            self.tokenizer.pad_id,
            self.tokenizer.mask_id,
            self.tokenizer.bos_id,
            self.tokenizer.unk_id,
        }
        ranked = np.argsort(-logits).tolist()
        if forced_token_id is not None:
            ranked.insert(0, forced_token_id)
        for candidate in dict.fromkeys(int(token_id) for token_id in ranked):
            if candidate in blocked:
                continue
            text = self.tokenizer.decode([*prefix, candidate])
            if candidate == self.tokenizer.eos_id:
                if self._certify(text) is not None:
                    return candidate
                continue
            try:
                if dfa_admits_token(self.tokenizer, prefix, candidate):
                    return candidate
            except Exception:  # noqa: BLE001
                continue
        return None

    def generate(
        self,
        prompt: str,
        gold: Any | None = None,
        max_len: int | None = None,
        grammar_constrained: bool | None = None,
        design_md: str | None = None,
    ) -> str:
        del gold
        use_grammar = (
            bool(self.config.grammar_constrained)
            if grammar_constrained is None
            else grammar_constrained
        )
        requested = max_len or self.gen_len
        length = min(
            max(8, int(requested)),
            int(self.config.max_target_len),
            max(8, int(self.config.grammar_ltr_max_tokens)),
        )
        context, ctx_pad_mask = self._encode_context(
            self._format_context(prompt, design_md)
        )
        ids = np.full((1, length), self.tokenizer.mask_id, dtype=np.int64)
        ids[0, 0] = self.tokenizer.bos_id
        structural = sorted(structural_token_ids(self.tokenizer))
        bias = float(getattr(self.config, "structural_bias", 0.0) or 0.0)

        for position in range(1, length):
            logits = self.denoiser_session.run(
                ["logits"],
                {
                    "noisy_ids": ids,
                    "context": context,
                    "ctx_pad_mask": ctx_pad_mask,
                },
            )[0][0, position].copy()
            if use_grammar and bias and structural:
                logits[structural] += bias
            prefix = ids[0, :position].tolist()
            if use_grammar:
                forced = (
                    force_emit_token_id(self.tokenizer, prefix)
                    if bool(getattr(self.config, "grammar_fastpath", True))
                    else None
                )
                choice = self._pick_constrained_token(
                    logits,
                    prefix,
                    forced,
                )
            else:
                choice = int(logits.argmax())
            if choice is None:
                choice = self.tokenizer.eos_id
            ids[0, position] = choice
            if choice == self.tokenizer.eos_id:
                ids[0, position + 1 :] = self.tokenizer.pad_id
                break

        text = self.tokenizer.decode(ids[0].tolist()).strip()
        if use_grammar:
            # The web harness owns retry and browser fallback. Returning a
            # canned valid template here would incorrectly label a failed
            # model decode as a successful real-model attempt.
            return self._certify(text) or text
        return text
