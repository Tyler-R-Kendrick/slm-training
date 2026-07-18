"""LoRA causal-LM plug-in with cached OpenUI grammar token masks."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.parser import stream_check, validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.distill.trace_store import decode_config_hash
from slm_training.lineage.records import content_sha
from slm_training.lineage.tracks import CAUSAL_LORA_RECIPE
from slm_training.models.causal_trace import (
    AllowedIds,
    CausalTracedGeneration,
    CausalTraceIdentity,
    CausalTraceWriter,
    GeneratedOutcome,
    TracePolicy,
    capture_raw_steps,
    fold_policy_identity,
)


@dataclass(frozen=True)
class CausalLMOpenUIConfig:
    base_model_id: str
    base_model_revision: str
    max_length: int = 512
    device: str = "cpu"
    local_files_only: bool = False


class CausalLMOpenUIPlugin:
    """Transformers/PEFT adapter sharing the repository OpenUI validator."""

    def __init__(
        self, model: Any, tokenizer: Any, config: CausalLMOpenUIConfig
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._grammar_mask_cache: dict[tuple[int, ...], tuple[int, ...]] = {}

    @classmethod
    def from_pretrained(cls, config: CausalLMOpenUIConfig) -> CausalLMOpenUIPlugin:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional training extra
            raise RuntimeError(
                "install slm-training[hf] for the causal-LM track"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_id,
            revision=config.base_model_revision,
            local_files_only=config.local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            revision=config.base_model_revision,
            local_files_only=config.local_files_only,
        )
        model.to(config.device)
        return cls(model, tokenizer, config)

    def enable_lora(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional training extra
            raise RuntimeError("install slm-training[hf] for LoRA") from exc
        recipe = CAUSAL_LORA_RECIPE
        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=int(recipe["rank"]),
                lora_alpha=int(recipe["alpha"]),
                lora_dropout=float(recipe["dropout"]),
                target_modules=list(recipe["target_modules"]),
                task_type="CAUSAL_LM",
            ),
        )

    def train_sft(
        self,
        records: Iterable[ExampleRecord],
        output_dir: Path,
        *,
        target_token_budget: int,
        seed: int,
        resume_from_checkpoint: Path | None = None,
    ) -> dict[str, Any]:
        """Run the frozen LoRA recipe, restoring Trainer state only on resume."""
        try:
            from datasets import Dataset
            from transformers import EarlyStoppingCallback
            from trl import SFTConfig, SFTTrainer
        except ImportError as exc:  # pragma: no cover - optional training extra
            raise RuntimeError(
                "install slm-training[hf] for causal LoRA training"
            ) from exc
        rows = [{"text": self._training_text(record)} for record in records]
        if len(rows) < 2:
            raise ValueError("causal SFT requires at least two records")
        dataset = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=seed)
        recipe = CAUSAL_LORA_RECIPE
        effective_batch = int(recipe["effective_batch_size"])
        max_length = int(recipe["sequence_length"])
        max_steps = max(1, target_token_budget // (effective_batch * max_length))
        args = SFTConfig(
            output_dir=str(output_dir),
            max_length=max_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=effective_batch,
            learning_rate=float(recipe["learning_rate"]),
            lr_scheduler_type=str(recipe["scheduler"]),
            warmup_ratio=float(recipe["warmup_ratio"]),
            max_steps=max_steps,
            eval_strategy="steps",
            eval_steps=max(1, max_steps // 10),
            save_steps=max(1, max_steps // 10),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["trackio"],
            run_name=output_dir.name,
            seed=seed,
        )
        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        result = trainer.train(
            resume_from_checkpoint=(
                str(resume_from_checkpoint)
                if resume_from_checkpoint is not None
                else None
            )
        )
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(output_dir)
        return {"checkpoint": str(output_dir), "metrics": dict(result.metrics)}

    def artifact_identity(self) -> dict[str, str]:
        tokenizer_payload = getattr(self.tokenizer, "init_kwargs", {})
        return {
            "kind": "causal_lm_openui",
            "base_model_id": self.config.base_model_id,
            "base_model_revision": self.config.base_model_revision,
            "tokenizer_sha": content_sha(tokenizer_payload),
        }

    def compatibility_fingerprint(self) -> str:
        shapes = {
            name: tuple(value.shape) for name, value in self.model.state_dict().items()
        }
        return content_sha(
            {
                **self.artifact_identity(),
                "architecture": getattr(
                    self.model.config, "model_type", type(self.model).__name__
                ),
                "shapes": shapes,
            }
        )

    def forward(self, batch: list[ExampleRecord]) -> float:
        import torch

        texts = [self._training_text(record) for record in batch]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.model.device)
        output = self.model(**encoded, labels=encoded["input_ids"])
        return float(output.loss.detach().to(torch.float32).cpu().item())

    def generate(self, prompt: str, gold: ExampleRecord | None = None) -> str:
        del gold
        return self.generate_constrained(prompt)

    def generate_batch_requests(self, requests: list[GenerationRequest]) -> list[str]:
        return [self.generate_constrained(request.prompt) for request in requests]

    def generate_constrained(self, prompt: str, **kwargs: object) -> str:
        import torch

        system = "Return only one valid OpenUI program using placeholder content."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
        else:
            input_ids = self.tokenizer(f"{system}\n{prompt}\n", return_tensors="pt")[
                "input_ids"
            ].to(self.model.device)
        start = int(input_ids.shape[1])

        def allowed(_batch_id: int, ids: Any) -> list[int]:
            prefix = tuple(int(value) for value in ids[start:].tolist())
            return list(self._allowed_ids(prefix))

        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                max_new_tokens=int(
                    kwargs.get("max_new_tokens", self.config.max_length)
                ),
                do_sample=False,
                prefix_allowed_tokens_fn=allowed,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            output[0, start:], skip_special_tokens=True
        ).strip()
        program = validate(text)
        return (program.serialized or text).strip()

    def _allowed_ids(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        cached = self._grammar_mask_cache.get(prefix)
        if cached is not None:
            return cached
        eos = int(self.tokenizer.eos_token_id)
        prefix_text = self.tokenizer.decode(prefix, skip_special_tokens=True)
        allowed: list[int] = []
        for token_id in range(len(self.tokenizer)):
            if token_id == eos:
                try:
                    validate(prefix_text)
                except Exception:  # noqa: BLE001
                    continue
                allowed.append(token_id)
                continue
            trial = self.tokenizer.decode((*prefix, token_id), skip_special_tokens=True)
            if trial == prefix_text:
                continue
            try:
                status = stream_check(trial)
            except Exception:  # noqa: BLE001
                continue
            if not status.hard_error:
                allowed.append(token_id)
        if not allowed:
            allowed = [eos]
        result = tuple(allowed)
        self._grammar_mask_cache[prefix] = result
        return result

    def _encode_prompt(self, prompt: str) -> Any:
        """Encode the OpenUI chat prompt to input ids (shared by the traced paths)."""
        system = "Return only one valid OpenUI program using placeholder content."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
        return self.tokenizer(f"{system}\n{prompt}\n", return_tensors="pt")[
            "input_ids"
        ].to(self.model.device)

    def active_adapter_identity(self) -> str:
        """Content digest of the active adapter tensors, or "" when none are active.

        Hashing the tensor *values* (not just their names) means any mutation of an
        adapter weight changes the folded ``policy_checkpoint_sha`` — so a trace
        captured under one adapter cannot silently load as another.
        """
        import torch

        adapters = [
            (name, param)
            for name, param in self.model.named_parameters()
            if "lora" in name.lower()
        ]
        if not adapters:
            return ""
        return content_sha(
            {
                name: param.detach().to(torch.float64).cpu().flatten().tolist()
                for name, param in sorted(adapters, key=lambda item: item[0])
            }
        )

    def capture_identity(self, *, group_id: str, context_text: str) -> CausalTraceIdentity:
        """Build the trajectory identity stamped onto every captured causal state.

        Base and adapter identity are folded into a single policy fingerprint because
        the shared ``DecisionStateV2`` schema carries no dedicated adapter field, so an
        adapter-enabled and adapter-disabled capture receive different state identities.
        """
        adapter = self.active_adapter_identity()
        return CausalTraceIdentity(
            group_id=group_id,
            context_text=context_text,
            policy_checkpoint_sha=fold_policy_identity(
                self.compatibility_fingerprint(), adapter
            ),
            tokenizer_sha=self.artifact_identity()["tokenizer_sha"],
            decode_config_hash=decode_config_hash(getattr(self.model, "config", {})),
            base_model_revision=self.config.base_model_revision,
            adapter_identity=adapter,
        )

    def generate_constrained_traced(
        self,
        prompt: str,
        *,
        group_id: str,
        policy: TracePolicy | None = None,
        trace_writer: CausalTraceWriter | None = None,
        max_new_tokens: int | None = None,
        allowed_ids_fn: AllowedIds | None = None,
    ) -> CausalTracedGeneration:
        """Greedy constrained generation that captures exact per-step decision evidence.

        Unlike :meth:`generate_constrained` (unchanged and trace-free by default), this
        drives a per-step loop so the raw pre-mask logits, raw argmax, legal set, and
        constrained selection are all recoverable from integer prefix ids. The stored
        ``context_ids`` are the full prefix (prompt + generated suffix), so a consumer
        can replay ``model(context_ids).logits[:, -1, :]`` exactly. ``allowed_ids_fn``
        overrides the grammar legal-set seam (defaulting to :meth:`_allowed_ids`).
        """
        import torch

        input_ids = self._encode_prompt(prompt)
        prompt_row = tuple(int(value) for value in input_ids[0].tolist())
        prompt_len = len(prompt_row)
        device = self.model.device

        def forward_logits(prefix: tuple[int, ...]) -> list[float]:
            row = torch.tensor([list(prefix)], device=device)
            with torch.inference_mode():
                logits = self.model(row).logits[0, -1, :]
            return logits.to(torch.float32).cpu().tolist()

        def grammar_allowed(prefix: tuple[int, ...]) -> tuple[int, ...]:
            return self._allowed_ids(tuple(int(token) for token in prefix[prompt_len:]))

        result = capture_raw_steps(
            forward_logits=forward_logits,
            allowed_ids=allowed_ids_fn or grammar_allowed,
            eos_id=int(self.tokenizer.eos_token_id),
            max_new_tokens=int(
                max_new_tokens if max_new_tokens is not None else self.config.max_length
            ),
            initial_prefix=prompt_row,
            policy=policy,
        )
        text = self.tokenizer.decode(
            result.generated_token_ids, skip_special_tokens=True
        ).strip()
        try:
            program = validate(text)
            final_text = (program.serialized or text).strip()
            valid = True
        except Exception:  # noqa: BLE001 - honest: an unfinished decode is not valid
            final_text = text
            valid = False
        if trace_writer is not None:
            trace_writer.record_all(result)
        return CausalTracedGeneration(text=final_text, result=result, valid=valid)

    def replay_causal_action(
        self,
        state: Any,
        forced_action_id: int,
        continuation_seed: int,
        generation_config: dict[str, Any] | None = None,
        *,
        allowed_ids_fn: AllowedIds | None = None,
    ) -> GeneratedOutcome:
        """Replay a forced first action on the exact stored prefix, then continue.

        The forced action is applied to ``state.context_ids`` (the exact integer prefix
        recovered without re-encoding text); continuation uses the deterministic
        constrained policy, so the same seed/config reproduces the outcome. No judge
        runs here — the pre-judge outcome is returned for the shared counterfactual
        owner to score.
        """
        import torch

        if state.context_ids is None:
            raise ValueError("causal replay requires a stored integer prefix (context_ids)")
        if int(forced_action_id) not in state.legal_action_ids:
            raise ValueError("forced causal action is not legal for the stored state")
        prompt_len = len(state.context_ids) - int(state.decision_position)
        if prompt_len < 0:
            raise ValueError("stored decision position exceeds the recorded prefix length")
        base_prefix = tuple(int(token) for token in state.context_ids) + (
            int(forced_action_id),
        )
        device = self.model.device
        config = generation_config or {}

        def forward_logits(prefix: tuple[int, ...]) -> list[float]:
            row = torch.tensor([list(prefix)], device=device)
            with torch.inference_mode():
                logits = self.model(row).logits[0, -1, :]
            return logits.to(torch.float32).cpu().tolist()

        def grammar_allowed(prefix: tuple[int, ...]) -> tuple[int, ...]:
            return self._allowed_ids(tuple(int(token) for token in prefix[prompt_len:]))

        result = capture_raw_steps(
            forward_logits=forward_logits,
            allowed_ids=allowed_ids_fn or grammar_allowed,
            eos_id=int(self.tokenizer.eos_token_id),
            max_new_tokens=int(config.get("max_new_tokens", self.config.max_length)),
            initial_prefix=base_prefix,
        )
        # The stored prefix already holds the tokens generated before this decision;
        # keep them so a decision_position > 0 replay materializes the full program.
        generated_suffix = tuple(int(token) for token in state.context_ids[prompt_len:])
        generated = (*generated_suffix, int(forced_action_id), *result.generated_token_ids)
        raw_text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            program = validate(raw_text)
            canonical: str | None = (program.serialized or raw_text).strip()
        except Exception:  # noqa: BLE001 - unfinished continuation has no canonical form
            canonical = None
        return GeneratedOutcome(
            action_id=int(forced_action_id),
            continuation_seed=int(continuation_seed),
            finish_reason=result.stop_reason,
            raw_program=raw_text,
            canonical_program=canonical,
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "openui_plugin.json").write_text(
            json.dumps(self.artifact_identity(), indent=2) + "\n", encoding="utf-8"
        )

    def load(self, path: Path, *, trainable: bool = False) -> None:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install slm-training[hf] for LoRA") from exc
        self.model = PeftModel.from_pretrained(self.model, path, is_trainable=trainable)

    def load_parent_weights(self, path: Path) -> None:
        """Load parent adapter/model weights; optimizers are intentionally absent."""
        if path.is_dir():
            self.load(path, trainable=True)
            return
        import torch

        payload = torch.load(path, map_location=self.model.device, weights_only=True)
        state = (
            payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        )
        self.model.load_state_dict(state, strict=True)

    def export(self, path: Path, *, format: str = "onnx") -> tuple[Path, ...]:
        if format != "onnx":
            raise ValueError("causal export supports format='onnx'")
        path.mkdir(parents=True, exist_ok=False)
        model = (
            self.model.merge_and_unload()
            if hasattr(self.model, "merge_and_unload")
            else self.model
        )
        merged = path / ".merged-hf"
        raw_onnx = path / ".onnx-fp"
        model.save_pretrained(merged, safe_serialization=True)
        self.tokenizer.save_pretrained(merged)
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            from optimum.onnxruntime import ORTModelForCausalLM
        except ImportError as exc:  # pragma: no cover - optional export extra
            raise RuntimeError(
                "install slm-training[hf] with optimum for causal ONNX export"
            ) from exc
        exported = ORTModelForCausalLM.from_pretrained(merged, export=True)
        exported.save_pretrained(raw_onnx)
        self.tokenizer.save_pretrained(path)
        artifacts: list[Path] = []
        for raw in raw_onnx.glob("*.onnx"):
            target = path / f"{raw.stem}.int8.onnx"
            quantize_dynamic(raw, target, weight_type=QuantType.QInt8)
            final = path / raw.name
            target.replace(final)
            artifacts.append(final)
        for name in ("config.json", "generation_config.json"):
            source = merged / name
            if source.exists():
                shutil.copy2(source, path / name)
                artifacts.append(path / name)
        artifacts.extend(
            item for item in path.iterdir() if item.is_file() and item not in artifacts
        )
        shutil.rmtree(merged)
        shutil.rmtree(raw_onnx)
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        if size > 1_000_000_000:
            raise ValueError(f"export exceeds 1GB: {size} bytes")
        return tuple(sorted(set(artifacts)))

    def _training_text(self, record: ExampleRecord) -> str:
        return f"Return only valid OpenUI.\nUser: {record.prompt}\nAssistant:\n{record.openui}"

    @property
    def architecture_sha(self) -> str:
        raw = json.dumps(
            getattr(self.model.config, "to_dict", lambda: {})(), sort_keys=True
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
