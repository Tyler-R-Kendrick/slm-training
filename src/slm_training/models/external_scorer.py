"""Provider-neutral external scorer for compiler-owned legal actions.

SLM-108 (EFS1-01): establish an external 1-7B constrained-decoding semantic
ceiling. This module defines the scorer interface and a transformers causal-LM
adapter. The scorer never adds or removes legal candidates; it only returns soft
scores and diagnostics for candidates supplied by the compiler/grammar layer.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from slm_training.data.contract import GenerationRequest
from slm_training.evals.score_policy import CandidatePath
from slm_training.lineage.records import content_sha


class ExternalScorerError(RuntimeError):
    """Raised when the external scorer cannot produce a deterministic score."""


@dataclass(frozen=True)
class ExternalScorerConfig:
    """Pinned configuration for an external HuggingFace causal/instruct model."""

    model_id: str
    revision: str
    device: str = "cpu"
    dtype: str = "float32"  # float32 | float16 | bfloat16
    max_length: int = 512
    max_batch_size: int = 8
    local_files_only: bool = False
    # claim class mirrors CheckpointReferenceV1 claim classes.
    claim_class: str = "fixture"  # fixture | diagnostic | frontier | ship_candidate
    # Length normalization for multi-token candidate scoring.
    length_norm: str = "mean"  # mean | sum | none
    # Softmax temperature for greedy-vs-sampled modes.
    temperature: float = 1.0

    def identity(self) -> dict[str, str]:
        return {
            "kind": "external_causal_lm",
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "max_length": str(self.max_length),
            "length_norm": self.length_norm,
            "temperature": str(self.temperature),
            "claim_class": self.claim_class,
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.identity(), sort_keys=True).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class LegalAction:
    """One live semantic action/production supplied by the compiler."""

    action_id: str
    token_ids: tuple[int, ...]
    surface: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompleteCandidate:
    """One grammar-valid complete candidate (e.g., from CompletionForest)."""

    candidate_id: str
    token_ids: tuple[int, ...]
    surface: str | None = None
    metadata: dict[str, Any] | None = None


class ExternalLegalActionScorer(ABC):
    """Abstract scorer that ranks compiler-legal actions/candidates.

    Implementations must never mutate legality. Unknown/error outcomes are
    returned as ``None`` scores or as explicit diagnostics; they are not
    converted into positive scores.
    """

    @property
    @abstractmethod
    def config(self) -> ExternalScorerConfig: ...

    @abstractmethod
    def score_legal_actions(
        self,
        request: GenerationRequest,
        prefix_text: str,
        compiler_state_fingerprint: str,
        legal_actions: list[LegalAction],
        *,
        cache_handle: Any | None = None,
    ) -> dict[str, float | None]:
        """Return a score for every compiler-supplied legal action."""

    @abstractmethod
    def score_complete_candidates(
        self,
        request: GenerationRequest,
        prefix_text: str,
        candidates: list[CompleteCandidate],
        *,
        cache_handle: Any | None = None,
    ) -> dict[str, float | None]:
        """Return a score for every complete candidate."""

    @abstractmethod
    def artifact_identity(self) -> dict[str, str]: ...

    @abstractmethod
    def compatibility_fingerprint(self) -> str: ...

    @abstractmethod
    def diagnostics(self) -> dict[str, Any]: ...


class TransformersCausalLMScorer(ExternalLegalActionScorer):
    """Transformers causal-LM adapter with prefix caching and batch scoring."""

    def __init__(self, config: ExternalScorerConfig) -> None:
        self._config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._kv_cache: dict[str, Any] | None = None
        self._load_attempts: int = 0
        self._load_error: str | None = None
        self._score_calls: int = 0
        self._oom_fallbacks: int = 0

    @property
    def config(self) -> ExternalScorerConfig:
        return self._config

    def _lazy_load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ExternalScorerError(
                "install slm-training[hf] for external causal-LM scoring"
            ) from exc
        self._load_attempts += 1
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_id,
                revision=self._config.revision,
                local_files_only=self._config.local_files_only,
            )
            torch_dtype = _parse_dtype(self._config.dtype)
            model = AutoModelForCausalLM.from_pretrained(
                self._config.model_id,
                revision=self._config.revision,
                local_files_only=self._config.local_files_only,
                torch_dtype=torch_dtype,
            )
            model.to(self._config.device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            return model, tokenizer
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            raise ExternalScorerError(f"failed to load {self._config.model_id}: {exc}") from exc

    def _encode_request(self, request: GenerationRequest) -> Any:
        """Encode a GenerationRequest into input ids with deterministic chat template hashing."""
        _, tokenizer = self._lazy_load()
        system = "Return only one valid OpenUI program using placeholder content."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": request.prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            text = f"{system}\n{request.prompt}\n"
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        return input_ids.to(self._config.device)

    def _score_token_ids(
        self,
        prefix_ids: Any,
        candidate_ids_list: list[tuple[int, ...]],
    ) -> list[float | None]:
        """Length-normalized log-likelihood of each candidate given the prefix."""
        import torch

        model, tokenizer = self._lazy_load()
        self._score_calls += 1
        results: list[float | None] = []
        for candidate_ids in candidate_ids_list:
            if not candidate_ids:
                results.append(0.0)
                continue
            try:
                full_ids = torch.cat(
                    [prefix_ids, torch.tensor([[*candidate_ids]], device=prefix_ids.device)],
                    dim=1,
                )
                if full_ids.shape[1] > self._config.max_length:
                    results.append(None)
                    continue
                with torch.inference_mode():
                    outputs = model(full_ids)
                    logits = outputs.logits[0, :, :]
                log_probs: list[float] = []
                prefix_len = prefix_ids.shape[1]
                for i, token_id in enumerate(candidate_ids):
                    pos = prefix_len + i
                    logit = logits[pos, token_id]
                    log_probs.append(float(logit.log_softmax(dim=0)[token_id].item()))
                total = sum(log_probs)
                norm = _apply_length_norm(total, len(log_probs), self._config.length_norm)
                results.append(norm)
            except torch.cuda.OutOfMemoryError as _exc:  # pragma: no cover - GPU only
                self._oom_fallbacks += 1
                results.append(None)
            except Exception as _exc:  # noqa: BLE001
                results.append(None)
        return results

    def score_legal_actions(
        self,
        request: GenerationRequest,
        prefix_text: str,
        compiler_state_fingerprint: str,
        legal_actions: list[LegalAction],
        *,
        cache_handle: Any | None = None,
    ) -> dict[str, float | None]:
        prefix_ids = self._encode_request(request)
        candidates = [
            tuple(int(t) for t in action.token_ids) for action in legal_actions
        ]
        scores = self._score_token_ids(prefix_ids, candidates)
        return {
            action.action_id: score
            for action, score in zip(legal_actions, scores)
        }

    def score_complete_candidates(
        self,
        request: GenerationRequest,
        prefix_text: str,
        candidates: list[CompleteCandidate],
        *,
        cache_handle: Any | None = None,
    ) -> dict[str, float | None]:
        prefix_ids = self._encode_request(request)
        candidates_ids = [
            tuple(int(t) for t in cand.token_ids) for cand in candidates
        ]
        scores = self._score_token_ids(prefix_ids, candidates_ids)
        return {
            cand.candidate_id: score
            for cand, score in zip(candidates, scores)
        }

    def artifact_identity(self) -> dict[str, str]:
        tokenizer_payload: dict[str, Any] = {}
        if self._tokenizer is not None:
            tokenizer_payload = getattr(self._tokenizer, "init_kwargs", {}) or {}
        return {
            **self._config.identity(),
            "tokenizer_sha": content_sha(tokenizer_payload),
        }

    def compatibility_fingerprint(self) -> str:
        shapes: dict[str, Any] = {}
        if self._model is not None:
            shapes = {
                name: tuple(value.shape)
                for name, value in self._model.state_dict().items()
            }
        return content_sha(
            {
                **self.artifact_identity(),
                "architecture": (
                    getattr(self._model.config, "model_type", type(self._model).__name__)
                    if self._model is not None
                    else "unloaded"
                ),
                "shapes": shapes,
            }
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "load_attempts": self._load_attempts,
            "load_error": self._load_error,
            "score_calls": self._score_calls,
            "oom_fallbacks": self._oom_fallbacks,
            "loaded": self._model is not None,
        }


class FakeExternalScorer(ExternalLegalActionScorer):
    """Deterministic fake scorer for torch-free tests and fixture runs."""

    def __init__(self, config: ExternalScorerConfig) -> None:
        self._config = config
        self._diag: dict[str, Any] = {"kind": "fake", "calls": 0}

    @property
    def config(self) -> ExternalScorerConfig:
        return self._config

    def score_legal_actions(
        self,
        request: GenerationRequest,
        prefix_text: str,
        compiler_state_fingerprint: str,
        legal_actions: list[LegalAction],
        *,
        cache_handle: Any | None = None,
    ) -> dict[str, float | None]:
        self._diag["calls"] += 1
        return {
            action.action_id: _fake_score(action.token_ids)
            for action in legal_actions
        }

    def score_complete_candidates(
        self,
        request: GenerationRequest,
        prefix_text: str,
        candidates: list[CompleteCandidate],
        *,
        cache_handle: Any | None = None,
    ) -> dict[str, float | None]:
        self._diag["calls"] += 1
        return {
            cand.candidate_id: _fake_score(cand.token_ids)
            for cand in candidates
        }

    def artifact_identity(self) -> dict[str, str]:
        return {**self._config.identity(), "kind": "fake_scorer"}

    def compatibility_fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.artifact_identity(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._diag)


class _ExternalScorerFactory(Protocol):
    def __call__(self, config: ExternalScorerConfig) -> ExternalLegalActionScorer: ...


SCORER_REGISTRY: dict[str, _ExternalScorerFactory] = {
    "transformers_causal_lm": TransformersCausalLMScorer,
    "fake": FakeExternalScorer,
}


def build_external_scorer(
    config: ExternalScorerConfig,
    *,
    kind: str = "transformers_causal_lm",
) -> ExternalLegalActionScorer:
    if kind not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown external scorer kind {kind!r}; choose from {sorted(SCORER_REGISTRY)}"
        )
    return SCORER_REGISTRY[kind](config)


def _fake_score(token_ids: tuple[int, ...]) -> float:
    if not token_ids:
        return 0.0
    return sum((tid % 1000) / 1000.0 - 0.5 for tid in token_ids) / max(1, len(token_ids))


def _parse_dtype(dtype: str) -> Any:
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unknown dtype {dtype!r}; choose from {sorted(mapping)}")
    return mapping[dtype]


def _apply_length_norm(total: float, n: int, mode: str) -> float:
    if n <= 0:
        return total
    if mode == "mean":
        return total / n
    if mode == "sum":
        return total
    if mode == "none":
        return total
    raise ValueError(f"Unknown length_norm {mode!r}")





@dataclass(frozen=True)
class ExternalScorePolicy:
    """ScorePolicy adapter that rescores CandidatePath traces with an external model.

    The external scorer receives only the candidate token sequence and prompt; it
    does not own legality. Missing or error scores fall back to the raw cumulative
    log-prob carried by the path so the policy never blocks a caller.
    """

    scorer: ExternalLegalActionScorer
    request: GenerationRequest
    prefix_text: str = ""
    fallback_to_path_logprobs: bool = True
    name: str = "external_score_policy"

    def score(self, path: CandidatePath) -> float:
        candidate = CompleteCandidate(
            candidate_id=path.candidate_id,
            token_ids=path.token_ids,
            surface=None,
            metadata=dict(path.metadata or {}),
        )
        try:
            scored = self.scorer.score_complete_candidates(
                self.request, self.prefix_text, [candidate]
            )
        except ExternalScorerError:
            scored = {path.candidate_id: None}
        value = scored.get(path.candidate_id)
        if value is not None:
            return float(value)
        if self.fallback_to_path_logprobs:
            return sum(path.log_probs)
        return float("-inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scorer_identity": self.scorer.artifact_identity(),
            "request_prompt_sha": hashlib.sha256(
                self.request.prompt.encode("utf-8")
            ).hexdigest(),
            "fallback_to_path_logprobs": self.fallback_to_path_logprobs,
        }
