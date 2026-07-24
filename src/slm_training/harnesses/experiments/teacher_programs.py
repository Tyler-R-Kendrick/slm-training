"""Fail-closed SLM-266 teacher-program request and admission contracts.

This owner freezes a provider-neutral generation manifest, offers explicitly
selected remote or local transports, and admits *already archived* raw responses
through the existing ProgramSpec, verifier, and leakage owners. A missing judge,
human audit, budget, campaign lock, or protected-split manifest is a rejection,
never an inferred pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.leakage import (
    find_leakage,
    fingerprint_openui,
    fingerprint_pair,
)
from slm_training.data.progspec.schema import ProgramSpec, emit_record
from slm_training.data.store import write_common_manifest
from slm_training.data.verify import (
    Gate,
    GateStatus,
    VerificationContext,
    verify_record,
)
from slm_training.dsl.schema import ExampleRecord, write_jsonl

REQUEST_SCHEMA = "teacher_program_request/v1"
GENERATION_SCHEMA = "teacher_program_generation_manifest/v1"
ADMISSION_SCHEMA = "teacher_program_admission/v1"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class AdmissionMode(StrEnum):
    DEEP_VERIFIED = "deep_verified"
    PARSE_ONLY = "parse_only"
    NO_CANONICAL_DEDUP = "no_canonical_dedup"


@dataclass(frozen=True)
class TeacherProgramRequestV1:
    """Source-intent request; never contains a target program or protected prompt."""

    request_id: str
    coverage_manifest_hash: str
    coverage_gap_ids: tuple[str, ...]
    allowed_components: tuple[str, ...]
    output_kind: str
    protected_exclusion_manifest_hash: str
    template_hash: str
    seed: int
    schema_version: str = REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if not all(
            (
                self.request_id,
                self.coverage_manifest_hash,
                self.output_kind,
                self.protected_exclusion_manifest_hash,
                self.template_hash,
            )
        ):
            raise ValueError("teacher request fields must be non-empty")
        if not self.coverage_gap_ids or not self.allowed_components:
            raise ValueError(
                "teacher request requires coverage gaps and allowed components"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "coverage_gap_ids": list(self.coverage_gap_ids),
            "allowed_components": list(self.allowed_components),
        }


@dataclass(frozen=True)
class TeacherProgramGenerationManifestV1:
    """Frozen no-spend manifest; an executor must enforce its caps separately."""

    manifest_id: str
    provider: str
    model: str
    revision: str
    request_ids: tuple[str, ...]
    max_dollars: float
    max_input_tokens: int
    max_output_tokens: int
    protected_exclusion_manifest_hash: str
    input_cost_per_1k_usd: float | None = None
    output_cost_per_1k_usd: float | None = None
    max_attempts: int = 1
    campaign_manifest_sha256: str | None = None
    schema_version: str = GENERATION_SCHEMA

    def __post_init__(self) -> None:
        if not all(
            (
                self.manifest_id,
                self.provider,
                self.model,
                self.revision,
                self.protected_exclusion_manifest_hash,
            )
        ):
            raise ValueError("generation manifest identity fields must be non-empty")
        if not self.request_ids:
            raise ValueError("generation manifest requires at least one request")
        if min(self.max_dollars, self.max_input_tokens, self.max_output_tokens) <= 0:
            raise ValueError("generation manifest requires positive hard budget caps")
        if (self.input_cost_per_1k_usd is None) != (
            self.output_cost_per_1k_usd is None
        ):
            raise ValueError("price schedule must include both input and output rates")
        if (
            self.input_cost_per_1k_usd is not None
            and min(self.input_cost_per_1k_usd, self.output_cost_per_1k_usd or 0.0) < 0
        ):
            raise ValueError("price schedule cannot be negative")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.campaign_manifest_sha256 is not None and (
            len(self.campaign_manifest_sha256) != 64
            or any(
                char not in "0123456789abcdef" for char in self.campaign_manifest_sha256
            )
        ):
            raise ValueError("campaign_manifest_sha256 must be a lowercase sha256")

    @property
    def manifest_hash(self) -> str:
        return _hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "request_ids": list(self.request_ids)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TeacherProgramGenerationManifestV1:
        return cls(
            manifest_id=str(data["manifest_id"]),
            provider=str(data["provider"]),
            model=str(data["model"]),
            revision=str(data["revision"]),
            request_ids=tuple(str(value) for value in data["request_ids"]),
            max_dollars=float(data["max_dollars"]),
            max_input_tokens=int(data["max_input_tokens"]),
            max_output_tokens=int(data["max_output_tokens"]),
            protected_exclusion_manifest_hash=str(
                data["protected_exclusion_manifest_hash"]
            ),
            input_cost_per_1k_usd=_optional_float(data.get("input_cost_per_1k_usd")),
            output_cost_per_1k_usd=_optional_float(data.get("output_cost_per_1k_usd")),
            max_attempts=int(data.get("max_attempts", 1)),
            campaign_manifest_sha256=_optional_str(
                data.get("campaign_manifest_sha256")
            ),
        )


@dataclass(frozen=True)
class TeacherProgramExecutionRequestV1:
    """A public source-intent prompt and declared worst-case token bounds."""

    request_id: str
    intent: str
    max_input_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        if not self.request_id or not self.intent.strip():
            raise ValueError("execution request id and intent must be non-empty")
        if min(self.max_input_tokens, self.max_output_tokens) < 1:
            raise ValueError("execution request token bounds must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TeacherProgramExecutionRequestV1:
        return cls(
            request_id=str(data["request_id"]),
            intent=str(data["intent"]),
            max_input_tokens=int(data["max_input_tokens"]),
            max_output_tokens=int(data["max_output_tokens"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TeacherBudgetExceeded(RuntimeError):
    """Raised before a provider call that could exceed the frozen budget."""


TeacherProgramTransport = Callable[
    [TeacherProgramExecutionRequestV1], Mapping[str, Any]
]


@dataclass(frozen=True)
class TeacherProgramExecutionResult:
    completed_request_ids: tuple[str, ...]
    skipped_request_ids: tuple[str, ...]
    attempt_artifact_sha256s: tuple[str, ...]
    spent_input_tokens: int
    spent_output_tokens: int
    spent_dollars: float


class TeacherProgramExecutor:
    """Bounded provider execution with content-addressed raw evidence.

    ``transport`` is injected so tests and non-OpenAI providers remain offline.
    A provider response must report token usage; missing usage is never inferred.
    Exceptions are archived and fail closed rather than being silently retried,
    because an unknown bill cannot safely be charged against a hard cap.
    """

    def __init__(
        self,
        manifest: TeacherProgramGenerationManifestV1,
        archive: CampaignStore,
    ) -> None:
        if (
            manifest.input_cost_per_1k_usd is None
            or manifest.output_cost_per_1k_usd is None
        ):
            raise ValueError("provider execution requires a pinned price schedule")
        if not manifest.campaign_manifest_sha256:
            raise ValueError(
                "provider execution requires a locked campaign manifest sha256"
            )
        self.manifest = manifest
        self.archive = archive

    def _prior_state(self) -> tuple[set[str], int, int, float]:
        completed: set[str] = set()
        input_tokens = output_tokens = 0
        dollars = 0.0
        for event in self.archive.verify_event_chain():
            detail = event.get("detail") or {}
            if (
                str(detail.get("generation_manifest_hash", ""))
                != self.manifest.manifest_hash
            ):
                continue
            request_id = str(detail.get("request_id", ""))
            if event.get("event_type") == "teacher_program_completed":
                completed.add(request_id)
            if event.get("event_type") == "teacher_program_attempted":
                usage = detail.get("usage") or {}
                input_tokens += int(usage.get("input_tokens", 0))
                output_tokens += int(usage.get("output_tokens", 0))
                dollars += float(detail.get("cost_usd", 0.0))
        return completed, input_tokens, output_tokens, dollars

    def _assert_budget(
        self,
        request: TeacherProgramExecutionRequestV1,
        *,
        input_tokens: int,
        output_tokens: int,
        dollars: float,
    ) -> None:
        assert self.manifest.input_cost_per_1k_usd is not None
        assert self.manifest.output_cost_per_1k_usd is not None
        projected_input = input_tokens + request.max_input_tokens
        projected_output = output_tokens + request.max_output_tokens
        projected_dollars = (
            dollars
            + (
                request.max_input_tokens * self.manifest.input_cost_per_1k_usd
                + request.max_output_tokens * self.manifest.output_cost_per_1k_usd
            )
            / 1000
        )
        if (
            projected_input > self.manifest.max_input_tokens
            or projected_output > self.manifest.max_output_tokens
            or projected_dollars > self.manifest.max_dollars
        ):
            raise TeacherBudgetExceeded(
                f"frozen teacher budget blocks request {request.request_id} before provider call"
            )

    def execute(
        self,
        requests: Iterable[TeacherProgramExecutionRequestV1],
        transport: TeacherProgramTransport,
    ) -> TeacherProgramExecutionResult:
        requests = tuple(sorted(requests, key=lambda request: request.request_id))
        if tuple(request.request_id for request in requests) != tuple(
            sorted(self.manifest.request_ids)
        ):
            raise ValueError(
                "execution request ids must exactly match the frozen manifest"
            )
        completed, input_tokens, output_tokens, dollars = self._prior_state()
        completed_now: list[str] = []
        skipped: list[str] = []
        artifacts: list[str] = []
        for request in requests:
            if request.request_id in completed:
                skipped.append(request.request_id)
                continue
            self._assert_budget(
                request,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                dollars=dollars,
            )
            request_artifact = self.archive.write_artifact(
                "teacher_program_requests",
                {
                    "generation_manifest_hash": self.manifest.manifest_hash,
                    "campaign_manifest_sha256": self.manifest.campaign_manifest_sha256,
                    "request": request.to_dict(),
                },
            )
            started = time.monotonic()
            try:
                response = dict(transport(request))
                usage = _provider_usage(response)
                if (
                    usage["input_tokens"] > request.max_input_tokens
                    or usage["output_tokens"] > request.max_output_tokens
                ):
                    raise ValueError(
                        "provider usage exceeded declared request token bound"
                    )
                cost = _cost(usage, self.manifest)
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
                dollars += cost
                attempt = {
                    "generation_manifest_hash": self.manifest.manifest_hash,
                    "campaign_manifest_sha256": self.manifest.campaign_manifest_sha256,
                    "request_id": request.request_id,
                    "request_artifact_sha256": request_artifact.stem,
                    "raw_response": response,
                    "usage": usage,
                    "cost_usd": cost,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                }
                artifact = self.archive.write_artifact(
                    "teacher_program_attempts", attempt
                )
                artifacts.append(artifact.stem)
                self.archive.append_event(
                    "teacher_program_attempted",
                    experiment_id=self.manifest.manifest_id,
                    status="success",
                    artifact_sha256=artifact.stem,
                    detail={
                        "generation_manifest_hash": self.manifest.manifest_hash,
                        "request_id": request.request_id,
                        "usage": usage,
                        "cost_usd": cost,
                    },
                )
                self.archive.append_event(
                    "teacher_program_completed",
                    experiment_id=self.manifest.manifest_id,
                    status="completed",
                    artifact_sha256=artifact.stem,
                    detail={
                        "generation_manifest_hash": self.manifest.manifest_hash,
                        "request_id": request.request_id,
                    },
                )
                completed_now.append(request.request_id)
            except Exception as exc:
                artifact = self.archive.write_artifact(
                    "teacher_program_attempts",
                    {
                        "generation_manifest_hash": self.manifest.manifest_hash,
                        "campaign_manifest_sha256": self.manifest.campaign_manifest_sha256,
                        "request_id": request.request_id,
                        "request_artifact_sha256": request_artifact.stem,
                        "error": type(exc).__name__,
                        "latency_ms": round((time.monotonic() - started) * 1000),
                    },
                )
                self.archive.append_event(
                    "teacher_program_attempted",
                    experiment_id=self.manifest.manifest_id,
                    status="error",
                    artifact_sha256=artifact.stem,
                    detail={
                        "generation_manifest_hash": self.manifest.manifest_hash,
                        "request_id": request.request_id,
                        "usage": {},
                        "cost_usd": 0.0,
                    },
                )
                raise
        return TeacherProgramExecutionResult(
            completed_request_ids=tuple(completed_now),
            skipped_request_ids=tuple(skipped),
            attempt_artifact_sha256s=tuple(artifacts),
            spent_input_tokens=input_tokens,
            spent_output_tokens=output_tokens,
            spent_dollars=dollars,
        )


def verify_teacher_generation_campaign_lock(
    manifest: TeacherProgramGenerationManifestV1,
    archive: CampaignStore,
) -> str:
    """Return the pre-execution lock digest matching this generation manifest."""
    if not manifest.campaign_manifest_sha256:
        raise ValueError("teacher generation requires a locked campaign manifest sha256")
    lock = archive.load_experiment_campaign(manifest.manifest_id)
    if lock.manifest.campaign_id != archive.campaign_id:
        raise ValueError("teacher generation campaign lock belongs to another archive")
    if lock.manifest_sha256 != manifest.campaign_manifest_sha256:
        raise ValueError("teacher generation campaign lock digest does not match manifest")
    return lock.manifest_sha256


class OpenAICompatibleTeacherTransport:
    """Small stdlib adapter for an explicitly configured OpenAI-compatible API."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key_env: str,
        model: str,
        system_prompt: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.model = model
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds

    def __call__(self, request: TeacherProgramExecutionRequestV1) -> Mapping[str, Any]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"required provider credential {self.api_key_env} is unset"
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": request.intent},
            ],
            "max_tokens": request.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        wire = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=wire,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.timeout_seconds
            ) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                f"provider request failed: {type(exc).__name__}"
            ) from exc
        usage = dict(raw.get("usage") or {})
        return {
            "raw": raw,
            "usage": {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            },
        }


class LocalTransformersTeacherTransport:
    """Pinned, offline-only causal-LM teacher transport.

    This is intentionally a generation adapter rather than an evaluator: the
    returned text remains untrusted raw evidence and must still flow through
    extraction, admission, and an independent judge.  ``local_files_only`` is
    fixed so a calibration can never silently download a model or call hosted
    inference.
    """

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        system_prompt: str,
        device: str = "cpu",
        dtype: str = "bfloat16",
    ) -> None:
        if not all((model, revision, system_prompt.strip())):
            raise ValueError("local teacher requires pinned model, revision, and prompt")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("local teacher dtype must be float32, float16, or bfloat16")
        self.model_id = model
        self.revision = revision
        self.system_prompt = system_prompt
        self.device = device
        self.dtype = dtype
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _lazy_load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "local teacher requires the torch and transformers extras"
            ) from exc
        torch_dtype = getattr(torch, self.dtype)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                local_files_only=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                local_files_only=True,
                torch_dtype=torch_dtype,
            )
            model.to(self.device)
            model.eval()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"failed to load local teacher {self.model_id}@{self.revision}: {type(exc).__name__}"
            ) from exc
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def __call__(self, request: TeacherProgramExecutionRequestV1) -> Mapping[str, Any]:
        import torch

        model, tokenizer = self._lazy_load()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": request.intent},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            input_ids = tokenizer(
                f"{self.system_prompt}\n{request.intent}\n",
                return_tensors="pt",
            )["input_ids"]
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)
        input_tokens = int(input_ids.shape[1])
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=request.max_output_tokens,
                do_sample=False,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
        output_ids = generated_ids[:, input_tokens:]
        output_tokens = int(output_ids.shape[1])
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        raw = {
            "provider": "local_transformers",
            "model": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "local_files_only": True,
            "response_text": text,
        }
        return {
            "raw": raw,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _provider_usage(response: Mapping[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise TypeError("provider response omitted usage")
    result = {key: int(usage[key]) for key in ("input_tokens", "output_tokens")}
    if min(result.values()) < 0:
        raise ValueError("provider usage cannot be negative")
    return result


def _cost(
    usage: Mapping[str, int], manifest: TeacherProgramGenerationManifestV1
) -> float:
    assert manifest.input_cost_per_1k_usd is not None
    assert manifest.output_cost_per_1k_usd is not None
    return (
        usage["input_tokens"] * manifest.input_cost_per_1k_usd
        + usage["output_tokens"] * manifest.output_cost_per_1k_usd
    ) / 1000


@dataclass(frozen=True)
class TeacherProgramCandidate:
    """One immutable raw response, with exactly one declared program payload."""

    candidate_id: str
    request_id: str
    response_id: str
    prompt: str
    program_payloads: tuple[str, ...]
    provider: str
    model: str
    revision: str
    generator_family: str
    judge_family: str | None
    coverage_gap_ids: tuple[str, ...]
    program_family_id: str
    lineage_id: str
    split_group_id: str
    provenance_complete: bool
    independent_judge_passed: bool | None = None
    judge_evidence: Mapping[str, Any] | None = None
    human_audit_passed: bool | None = None
    require_runtime: bool = False
    require_behavior: bool = False
    required_facts: tuple[str, ...] = ()
    forbidden_facts: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmissionResultV1:
    mode: AdmissionMode
    accepted: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, Any], ...]
    manifest_hash: str
    schema_version: str = ADMISSION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "accepted": list(self.accepted),
            "rejected": list(self.rejected),
            "accepted_n": len(self.accepted),
            "rejected_n": len(self.rejected),
            "manifest_hash": self.manifest_hash,
        }


@dataclass(frozen=True)
class TeacherRawArchiveRefV1:
    """Immutable durable archive pointer; raw provider I/O never enters Git data."""

    uri: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.uri or "://" not in self.uri:
            raise ValueError("raw archive uri must be an absolute URI")
        if len(self.manifest_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.manifest_sha256
        ):
            raise ValueError("raw archive manifest_sha256 must be a lowercase sha256")

    def to_dict(self) -> dict[str, str]:
        return {"uri": self.uri, "manifest_sha256": self.manifest_sha256}


def _required_gates(
    candidate: TeacherProgramCandidate, mode: AdmissionMode
) -> frozenset[Gate]:
    if mode is AdmissionMode.PARSE_ONLY:
        return frozenset({Gate.LEXICAL, Gate.GRAMMAR})
    gates = {
        Gate.LEXICAL,
        Gate.GRAMMAR,
        Gate.SCHEMA,
        Gate.REFERENCES,
        Gate.DATAFLOW,
        Gate.GROUNDING,
        Gate.CANONICAL,
        Gate.PROVENANCE,
        Gate.INDEPENDENT_JUDGE,
    }
    if candidate.require_runtime:
        gates.add(Gate.RUNTIME)
    if candidate.require_behavior:
        gates.add(Gate.BEHAVIOR)
    return frozenset(gates)


def _reject(
    candidate: TeacherProgramCandidate, reason: str, **detail: Any
) -> dict[str, Any]:
    return {"candidate_id": candidate.candidate_id, "reason": reason, **detail}


def _automatic_judge_passed(candidate: TeacherProgramCandidate) -> bool:
    """Validate a durable automatic cross-family judge record before G11."""
    evidence = candidate.judge_evidence
    if not isinstance(evidence, Mapping):
        raise TypeError("independent_judge_evidence_missing")
    required = (
        "candidate_id",
        "provider",
        "model_family",
        "approved",
        "raw_artifact_sha256",
        "prompt_sha256",
        "program_sha256",
    )
    missing = [key for key in required if key not in evidence]
    if missing:
        raise ValueError(f"independent_judge_evidence_missing:{','.join(missing)}")
    if str(evidence["candidate_id"]) != candidate.candidate_id:
        raise ValueError("independent_judge_candidate_mismatch")
    if str(evidence["provider"]) == candidate.provider:
        raise ValueError("independent_judge_provider_not_independent")
    if str(evidence["model_family"]) == candidate.generator_family:
        raise ValueError("independent_judge_family_not_independent")
    if (
        candidate.judge_family
        and str(evidence["model_family"]) != candidate.judge_family
    ):
        raise ValueError("independent_judge_family_mismatch")
    for key in ("raw_artifact_sha256", "prompt_sha256", "program_sha256"):
        value = str(evidence[key])
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"independent_judge_invalid_{key}")
    if (
        evidence["prompt_sha256"]
        != hashlib.sha256(candidate.prompt.encode("utf-8")).hexdigest()
    ):
        raise ValueError("independent_judge_prompt_hash_mismatch")
    if (
        evidence["program_sha256"]
        != hashlib.sha256(candidate.program_payloads[0].encode("utf-8")).hexdigest()
    ):
        raise ValueError("independent_judge_program_hash_mismatch")
    approved = evidence["approved"]
    if not isinstance(approved, bool):
        raise TypeError("independent_judge_approval_not_boolean")
    if (
        candidate.independent_judge_passed is not None
        and candidate.independent_judge_passed is not approved
    ):
        raise ValueError("independent_judge_boolean_disagrees_with_evidence")
    return approved


def _admit_one(
    candidate: TeacherProgramCandidate,
    *,
    mode: AdmissionMode,
    protected_fingerprints: Mapping[str, set[str]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if len(candidate.program_payloads) != 1:
        return None, _reject(
            candidate, "program_payload_count", count=len(candidate.program_payloads)
        )
    if (
        mode is not AdmissionMode.PARSE_ONLY
        and candidate.generator_family == candidate.judge_family
    ):
        return None, _reject(candidate, "generator_judge_not_independent")
    judge_passed = candidate.independent_judge_passed
    if mode is not AdmissionMode.PARSE_ONLY:
        try:
            judge_passed = _automatic_judge_passed(candidate)
        except (TypeError, ValueError) as exc:
            return None, _reject(candidate, str(exc))
    try:
        spec = ProgramSpec.from_openui(
            id=candidate.candidate_id,
            openui=candidate.program_payloads[0],
            facts=dict(candidate.facts),
            program_family_id=candidate.program_family_id,
            lineage_id=candidate.lineage_id,
            split_group_id=candidate.split_group_id,
            provenance={
                "teacher_provider": candidate.provider,
                "teacher_model": candidate.model,
                "teacher_revision": candidate.revision,
                "request_id": candidate.request_id,
                "response_id": candidate.response_id,
                "coverage_gap_ids": list(candidate.coverage_gap_ids),
            },
        )
    except (ValueError, TypeError) as exc:
        return None, _reject(candidate, "program_parse", detail=str(exc))
    record = emit_record(
        spec,
        prompt=candidate.prompt,
        task="generation",
        source="teacher",
        tier="Bronze",
        determinacy="teacher_generated",
        meta={
            "source_kind": "teacher",
            "provenance_complete": candidate.provenance_complete,
            "independent_judge_passed": judge_passed,
            "teacher_judge_evidence": dict(candidate.judge_evidence or {}),
            "human_audit_passed": candidate.human_audit_passed,
            "require_runtime": candidate.require_runtime,
            "require_behavior": candidate.require_behavior,
            "required_facts": list(candidate.required_facts),
            "forbidden_facts": list(candidate.forbidden_facts),
        },
    )
    leakage = find_leakage(record, dict(protected_fingerprints))
    if leakage:
        return None, _reject(candidate, "protected_split_leakage", leakage=leakage)
    report = verify_record(
        record,
        VerificationContext(
            source_kind="teacher",
            provenance_complete=candidate.provenance_complete,
            independent_judge_passed=judge_passed,
            human_audit_passed=candidate.human_audit_passed,
            require_runtime=candidate.require_runtime,
            require_behavior=candidate.require_behavior,
            required_facts=candidate.required_facts,
            forbidden_facts=candidate.forbidden_facts,
        ),
    )
    statuses = {result.gate: result.status for result in report.results}
    missing = sorted(
        gate.value
        for gate in _required_gates(candidate, mode)
        if statuses[gate] is not GateStatus.PASS
    )
    if missing:
        return None, _reject(
            candidate,
            "required_gate_not_pass",
            gates=missing,
            verification=report.to_dict(),
        )
    admission_tier = (
        None
        if mode is AdmissionMode.PARSE_ONLY
        else "Gold"
        if report.tier.value == "Gold"
        else "Silver"
    )
    return {
        "candidate_id": candidate.candidate_id,
        "record": record.to_dict(),
        "canonical_root_hash": fingerprint_openui(record.openui),
        "pair_hash": fingerprint_pair(record.prompt, record.openui),
        "verification": report.to_dict(),
        "admission_tier": admission_tier,
    }, None


def admit_teacher_programs(
    candidates: Iterable[TeacherProgramCandidate],
    *,
    mode: AdmissionMode | str,
    protected_fingerprints: Mapping[str, set[str]],
) -> AdmissionResultV1:
    """Admit one declared control axis from the identical raw candidate pool.

    Deep mode keeps one deterministic root representative; no-canonical-dedup
    keeps distinct records and only removes exact prompt/program byte pairs.
    """
    mode = AdmissionMode(mode)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        row, rejection = _admit_one(
            candidate, mode=mode, protected_fingerprints=protected_fingerprints
        )
        if row is None:
            rejected.append(rejection or _reject(candidate, "unknown"))
        else:
            accepted.append(row)
    seen: set[str] = set()
    materialized: list[dict[str, Any]] = []
    dedup_key = (
        "canonical_root_hash" if mode is AdmissionMode.DEEP_VERIFIED else "pair_hash"
    )
    for row in accepted:
        key = str(row[dedup_key])
        if key in seen:
            rejected.append(
                {
                    "candidate_id": row["candidate_id"],
                    "reason": f"duplicate_{dedup_key}",
                }
            )
        else:
            seen.add(key)
            materialized.append(row)
    payload = {
        "mode": mode.value,
        "accepted": materialized,
        "rejected": rejected,
    }
    return AdmissionResultV1(mode, tuple(materialized), tuple(rejected), _hash(payload))


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def materialize_teacher_admission(
    result: AdmissionResultV1,
    *,
    output_dir: Path | str,
    dataset_id: str,
    raw_archive: TeacherRawArchiveRefV1,
) -> dict[str, Any]:
    """Project only deep-verified rows into the canonical local train-data shape.

    The raw verifier tier remains in ``verification``. ``verification_tier`` is
    the separately auditable SLM-266 automated admission tier, so later strict
    curation can consume this snapshot without mistaking a teacher source for a
    fixture. Controls are deliberately non-materializable.
    """
    if result.mode is not AdmissionMode.DEEP_VERIFIED:
        raise ValueError(
            "only deep_verified teacher admission can materialize train data"
        )
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(f"teacher dataset already exists: {directory}")
    records: list[ExampleRecord] = []
    for row in result.accepted:
        admission_tier = str(row.get("admission_tier") or "")
        if admission_tier not in {"Gold", "Silver"}:
            raise ValueError("deep admission row lacks Gold/Silver admission tier")
        record = ExampleRecord.from_dict(dict(row["record"]))
        meta = {
            **record.meta,
            "tier": admission_tier,
            "verification_tier": admission_tier,
            "teacher_program_admission": {
                "schema_version": ADMISSION_SCHEMA,
                "admission_manifest_hash": result.manifest_hash,
                "candidate_id": row["candidate_id"],
                "canonical_root_hash": row["canonical_root_hash"],
                "pair_hash": row["pair_hash"],
                "admission_tier": admission_tier,
                "raw_archive": raw_archive.to_dict(),
            },
        }
        records.append(
            ExampleRecord(
                id=record.id,
                prompt=record.prompt,
                openui=record.openui,
                placeholders=list(record.placeholders),
                split=record.split,
                source=record.source,
                meta=meta,
                design_md=record.design_md,
                target_kind=record.target_kind,
                target_category=record.target_category,
                accepted_outputs=list(record.accepted_outputs),
            )
        )
    directory.mkdir(parents=True)
    records_path = directory / "records.jsonl"
    write_jsonl(records_path, records)
    _atomic_jsonl(directory / "rejected.jsonl", result.rejected)
    records_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "teacher_program_train_snapshot/v1",
        "dataset_id": dataset_id,
        "records": records_path.as_posix(),
        "rejected": (directory / "rejected.jsonl").as_posix(),
        "records_sha": records_sha256,
        "content_fingerprint": _hash(
            {
                "admission_manifest_hash": result.manifest_hash,
                "records_sha256": records_sha256,
                "raw_archive": raw_archive.to_dict(),
            }
        ),
        "admission": {
            "mode": result.mode.value,
            "manifest_hash": result.manifest_hash,
            "accepted_n": len(records),
            "rejected_n": len(result.rejected),
        },
        "raw_archive": raw_archive.to_dict(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return write_common_manifest(directory, kind="train", dataset_id=dataset_id)


__all__ = [
    "ADMISSION_SCHEMA",
    "GENERATION_SCHEMA",
    "REQUEST_SCHEMA",
    "AdmissionMode",
    "AdmissionResultV1",
    "OpenAICompatibleTeacherTransport",
    "TeacherBudgetExceeded",
    "TeacherProgramCandidate",
    "TeacherProgramExecutionRequestV1",
    "TeacherProgramExecutionResult",
    "TeacherProgramExecutor",
    "TeacherProgramGenerationManifestV1",
    "TeacherProgramRequestV1",
    "TeacherRawArchiveRefV1",
    "admit_teacher_programs",
    "materialize_teacher_admission",
    "verify_teacher_generation_campaign_lock",
]
