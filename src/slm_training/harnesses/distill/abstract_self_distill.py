"""On-policy abstract self-distillation collector (AP-020 / SLM-309).

Closes the warm-up train-inference gap: samples prompt-only abstract traces
from a live policy via AP-017's ``CausalLMOpenUIPlugin.generate_abstract_traced``
(SLM-304, ``models/abstract_decode.py``), independently re-verifies the raw
answer decode, and persists deterministic, resumable, deduplicated, sharded
records with complete policy/verifier provenance. Rejected samples are kept,
never dropped, so they remain auditable.

Independent verification matters here specifically because
``generate_abstract_traced`` always reports ``valid=True``: on a parse
failure it silently substitutes ``_certified_fallback()`` (a fixed trivial
program) rather than raising, so its own ``valid`` flag cannot be trusted as
an acceptance signal -- "the generating policy cannot be the sole semantic
judge" (per the ticket). This module re-decodes and re-validates the *raw*
``capture.answer.token_ids`` itself, independently of whatever the
generation path already decided.

This is a pure side channel: nothing here is called by any existing
generation or training path, so bottleneck-only (self-distillation
disabled) behavior is completely unaffected by this module's existence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from slm_training.dsl.parser import validate
from slm_training.models.abstract_decode import AbstractTraceCapture, AbstractTracedGeneration
from slm_training.models.causal_lm_openui import CausalLMOpenUIPlugin

DEFAULT_SHARD_COUNT = 16


def _trace_logprob(capture: AbstractTraceCapture) -> float | None:
    """Sum of every recorded per-token logprob (None entries are forward-free commits)."""
    logprobs = [
        logprob
        for phase in (capture.abstract, capture.answer)
        for logprob in phase.logprobs
        if logprob is not None
    ]
    return sum(logprobs) if logprobs else None


def _content_hash(
    policy_checkpoint_sha: str,
    prompt: str,
    abstract_token_ids: Sequence[int],
    answer_token_ids: Sequence[int],
) -> str:
    """Deterministic dedup key, mirroring data.leakage's fingerprint-set pattern."""
    payload = json.dumps(
        {
            "policy_checkpoint_sha": policy_checkpoint_sha,
            "prompt": prompt,
            "abstract_token_ids": list(abstract_token_ids),
            "answer_token_ids": list(answer_token_ids),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _shard_for(content_hash: str, num_shards: int) -> int:
    """Deterministic sha256-mod-modulus sharding, mirroring split_policy.py's convention."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    return int(content_hash, 16) % num_shards


def _verify_raw_answer(raw_text: str) -> tuple[bool, str]:
    """Independent verifier: the generating policy is never the sole judge."""
    try:
        validate(raw_text)
    except Exception as exc:  # noqa: BLE001 - any parse/schema failure is a rejection
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


@dataclass(frozen=True)
class SelfDistillRecordV1:
    """One collected on-policy abstract-trace record with complete provenance."""

    schema: str = "self_distill_record/v1"
    content_hash: str = ""
    shard: int = 0
    prompt: str = ""
    abstract_token_ids: tuple[int, ...] = ()
    answer_token_ids: tuple[int, ...] = ()
    raw_answer_text: str = ""
    policy_checkpoint_sha: str = ""
    codebook_version: int = 0
    plan_fingerprint: str = ""
    abstract_temperature: float = 0.0
    abstract_seed: int = 0
    answer_temperature: float = 0.0
    answer_seed: int = 0
    forced_end: bool = False
    abstract_termination: str = ""
    stop_reason: str = ""
    trace_logprob: float | None = None
    verifier_accepted: bool = False
    verifier_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "content_hash": self.content_hash,
            "shard": self.shard,
            "prompt": self.prompt,
            "abstract_token_ids": list(self.abstract_token_ids),
            "answer_token_ids": list(self.answer_token_ids),
            "raw_answer_text": self.raw_answer_text,
            "policy_checkpoint_sha": self.policy_checkpoint_sha,
            "codebook_version": self.codebook_version,
            "plan_fingerprint": self.plan_fingerprint,
            "abstract_temperature": self.abstract_temperature,
            "abstract_seed": self.abstract_seed,
            "answer_temperature": self.answer_temperature,
            "answer_seed": self.answer_seed,
            "forced_end": self.forced_end,
            "abstract_termination": self.abstract_termination,
            "stop_reason": self.stop_reason,
            "trace_logprob": self.trace_logprob,
            "verifier_accepted": self.verifier_accepted,
            "verifier_reason": self.verifier_reason,
        }


@dataclass
class CollectionSummary:
    """Yield/trace-length report for one collection run."""

    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicate: int = 0
    abstract_lengths: list[int] = field(default_factory=list)
    answer_lengths: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        count = len(self.abstract_lengths)
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "duplicate": self.duplicate,
            "mean_abstract_length": (sum(self.abstract_lengths) / count) if count else 0.0,
            "mean_answer_length": (sum(self.answer_lengths) / count) if count else 0.0,
        }


def build_self_distill_record(
    plugin: CausalLMOpenUIPlugin,
    prompt: str,
    generation: AbstractTracedGeneration,
    *,
    num_shards: int = DEFAULT_SHARD_COUNT,
) -> SelfDistillRecordV1:
    """Join AP-017's decode-side trace with AP-020's policy/verifier provenance.

    Independently re-decodes and re-validates ``capture.answer.token_ids``
    rather than trusting ``generation.valid`` (see module docstring).
    """
    capture = generation.capture
    raw_text = plugin.tokenizer.decode(
        capture.answer.token_ids, skip_special_tokens=True
    ).strip()
    accepted, reason = _verify_raw_answer(raw_text)
    policy_checkpoint_sha = plugin.compatibility_fingerprint()
    content_hash = _content_hash(
        policy_checkpoint_sha,
        prompt,
        capture.abstract.token_ids,
        capture.answer.token_ids,
    )
    abstract_config = plugin.config.abstract
    plan = abstract_config.plan if abstract_config is not None else None
    return SelfDistillRecordV1(
        content_hash=content_hash,
        shard=_shard_for(content_hash, num_shards),
        prompt=prompt,
        abstract_token_ids=tuple(capture.abstract.token_ids),
        answer_token_ids=tuple(capture.answer.token_ids),
        raw_answer_text=raw_text,
        policy_checkpoint_sha=policy_checkpoint_sha,
        codebook_version=plan.codebook_version if plan is not None else 0,
        plan_fingerprint=capture.plan_fingerprint,
        abstract_temperature=abstract_config.abstract.temperature
        if abstract_config is not None
        else 0.0,
        abstract_seed=abstract_config.abstract.seed if abstract_config is not None else 0,
        answer_temperature=abstract_config.answer.temperature
        if abstract_config is not None
        else 0.0,
        answer_seed=abstract_config.answer.seed if abstract_config is not None else 0,
        forced_end=capture.forced_end,
        abstract_termination=capture.abstract_termination,
        stop_reason=capture.stop_reason,
        trace_logprob=_trace_logprob(capture),
        verifier_accepted=accepted,
        verifier_reason=reason,
    )


def collect_on_policy_traces(
    plugin: CausalLMOpenUIPlugin,
    prompts: Sequence[str],
    *,
    num_shards: int = DEFAULT_SHARD_COUNT,
    seen_content_hashes: set[str] | None = None,
    generate_fn: Callable[[str], AbstractTracedGeneration] | None = None,
) -> tuple[list[SelfDistillRecordV1], CollectionSummary]:
    """Deterministically collect, verify, dedup, and shard on-policy abstract traces.

    ``seen_content_hashes`` supports resume: pass hashes already persisted
    from a prior run (see :func:`load_seen_hashes`) and matching prompts are
    skipped as duplicates rather than re-invoking the policy or re-appending
    a copy. Rejected (verifier-failed) records are still returned -- never
    silently dropped -- so they remain auditable.
    """
    if plugin.config.abstract is None:
        raise ValueError(
            "collect_on_policy_traces requires an enabled AbstractDecodeConfig "
            "(plugin.config.abstract); bottleneck-only plugins must not call this"
        )
    generate = generate_fn or plugin.generate_abstract_traced
    seen = set(seen_content_hashes or ())
    records: list[SelfDistillRecordV1] = []
    summary = CollectionSummary()
    for prompt in prompts:
        summary.attempted += 1
        generation = generate(prompt)
        record = build_self_distill_record(
            plugin, prompt, generation, num_shards=num_shards
        )
        if record.content_hash in seen:
            summary.duplicate += 1
            continue
        seen.add(record.content_hash)
        records.append(record)
        summary.abstract_lengths.append(len(record.abstract_token_ids))
        summary.answer_lengths.append(len(record.answer_token_ids))
        if record.verifier_accepted:
            summary.accepted += 1
        else:
            summary.rejected += 1
    return records, summary


def self_distill_training_batch(
    plugin: CausalLMOpenUIPlugin,
    prompt_token_ids: Sequence[int],
    record: SelfDistillRecordV1,
) -> dict[str, float | int]:
    """Train loss on abstract plus target tokens for one accepted record.

    Reuses AP-019's ``forward_with_segments`` (SLM-307,
    ``models/block_attention.py``): the loss mask already restricts credit
    to abstract+target positions, so this only needs to build the
    ``[prompt; abstract; target]`` segment ids -- there is no privileged
    plan at on-policy generation time, so the prompt segment is followed
    directly by abstract (no ``PRIVILEGED_PLAN`` positions), which the
    bottleneck mask already treats identically to plain causal order.
    """
    if not record.verifier_accepted:
        raise ValueError(
            "cannot build a training batch from a rejected record "
            f"(verifier_reason={record.verifier_reason!r})"
        )
    import torch

    from slm_training.models.block_attention import SegmentKind

    segment_ids = (
        [int(SegmentKind.PROMPT)] * len(prompt_token_ids)
        + [int(SegmentKind.ABSTRACT)] * len(record.abstract_token_ids)
        + [int(SegmentKind.TARGET)] * len(record.answer_token_ids)
    )
    input_ids = torch.tensor(
        [list(prompt_token_ids) + list(record.abstract_token_ids) + list(record.answer_token_ids)]
    )
    segment_tensor = torch.tensor([segment_ids], dtype=torch.long)
    return plugin.forward_with_segments(input_ids, segment_tensor)


def load_seen_hashes(path: Path) -> set[str]:
    """Load prior content hashes for resume (mirrors data.leakage's fingerprint sets)."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            seen.add(json.loads(line)["content_hash"])
    return seen


def append_records(path: Path, records: Sequence[SelfDistillRecordV1]) -> None:
    """Append-only JSONL persistence; never rewrites existing rows (mirrors TraceStore)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
