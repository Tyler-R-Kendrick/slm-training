"""On-policy abstract self-distillation collector (AP-020 / SLM-309).

Closes the warm-up train-inference gap: samples *prompt-only* abstract traces
through the AP-017 codebook-only abstract decoder
(:func:`slm_training.models.abstract_decode.capture_abstract_trace`, wired as
``CausalLMOpenUIPlugin.generate_abstract_traced``) and verifies each candidate
answer independently before it is ever eligible for training -- the
generating policy is never the sole semantic judge. Records are written
through the existing append-only :class:`~slm_training.harnesses.distill.trace_store.TraceStore`
using the same ``meta``/``labels``/``final`` shape
:mod:`slm_training.harnesses.distill.select` already filters and strata, so
downstream selection and SFT need no new plumbing.

Collection is:

* **deterministic** -- prompts are processed in the caller-supplied order; a
  ``shard_index``/``num_shards`` split selects a reproducible subset for
  parallel collection without inventing a new shard-file format.
* **resumable** -- every collected record's fingerprint (over the policy
  checkpoint, decode-config hash, and prompt) is checked against what the
  store already holds before generating, so a restarted run never re-samples
  a prompt it already collected under the same policy/config.
* **duplicate-free** -- a second fingerprint (policy + decode config + the
  generated *answer* text, independent of which prompt produced it) catches
  the same output recurring for a different prompt -- e.g. mode collapse onto
  a generic fallback -- and is never appended twice.
* **auditable on rejection** -- failed/rejected candidates are still written
  (with ``labels.accepted = False`` and a ``reject_reason``); nothing is
  discarded silently, matching :func:`~slm_training.harnesses.distill.select.corpus_label`'s
  existing "rejected" corpus.

When ``SelfDistillCollectConfig.enabled`` is ``False`` this is a pure no-op:
the store is untouched and ``generate`` is never called, so a bottleneck-only
training run (AP-019's masked SFT loss without self-distilled data) stays
byte-for-byte reproducible whether or not this collector exists.

Training on the collected ``[prompt; abstract; verified target]`` triples
reuses the AP-019 masked SFT loss primitive
(:mod:`slm_training.models.block_attention`) rather than a new trainer:
:func:`training_example_from_capture` builds the ``input_ids``/``segment_ids``
pair for ``CausalLMOpenUIPlugin.forward_with_segments`` directly from a
collected trace's :class:`~slm_training.models.abstract_decode.AbstractTraceCapture`.
No privileged-plan segment is used here (self-distillation traces carry no
separate plan channel), so this degenerates to AP-019's bottleneck mask with
the middle segment empty: plain causal attention, loss restricted to the
abstract + answer positions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from slm_training.evals.verifier_cascade import (
    Verdict,
    VerifierCascade,
    default_openui_cascade,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.distill.trace_store import TRACE_VERSION, TraceStore
from slm_training.models.abstract_decode import AbstractTraceCapture, AbstractTracedGeneration

__all__ = [
    "SOURCE_FAMILY",
    "CollectionSummary",
    "GenerateAbstractTrace",
    "SelfDistillCollectConfig",
    "collect_on_policy_traces",
    "segment_ids_for_capture",
    "training_example_from_capture",
]

SOURCE_FAMILY = "on_policy_abstract_self_distilled"

#: Injected policy callable: prompt text -> a captured abstract trace. Kept as
#: a callable (not a model type) so the collector itself stays torch-free and
#: unit-testable with fixture generations, mirroring ``abstract_decode.py``'s
#: injected-``forward_logits`` convention.
GenerateAbstractTrace = Callable[[str], AbstractTracedGeneration]


@dataclass(frozen=True)
class SelfDistillCollectConfig:
    """Default-on collector config; set ``enabled=False`` for the bottleneck-only control."""

    enabled: bool = True
    max_records: int | None = None
    shard_index: int = 0
    num_shards: int = 1
    reject_forced_end: bool = False
    min_abstract_tokens: int = 0

    def __post_init__(self) -> None:
        if self.num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        if not (0 <= self.shard_index < self.num_shards):
            raise ValueError("shard_index must be in [0, num_shards)")
        if self.max_records is not None and self.max_records < 0:
            raise ValueError("max_records must be non-negative")
        if self.min_abstract_tokens < 0:
            raise ValueError("min_abstract_tokens must be non-negative")


@dataclass(frozen=True)
class CollectionSummary:
    """Collection yield and trace-length distribution for one collection run."""

    candidate_count: int
    shard_count: int
    processed: int
    skipped_resumed: int
    duplicate: int
    accepted: int
    rejected: int
    trace_length_counts: dict[int, int] = field(default_factory=dict)

    @property
    def yield_rate(self) -> float:
        return self.accepted / self.processed if self.processed else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "shard_count": self.shard_count,
            "processed": self.processed,
            "skipped_resumed": self.skipped_resumed,
            "duplicate": self.duplicate,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "trace_length_counts": dict(self.trace_length_counts),
            "yield_rate": self.yield_rate,
        }


def _prompt_fingerprint(
    *, policy_checkpoint_sha: str, decode_config_hash: str, prompt: str
) -> str:
    return content_sha(
        {
            "family": SOURCE_FAMILY,
            "policy_checkpoint_sha": policy_checkpoint_sha,
            "decode_config_hash": decode_config_hash,
            "prompt": prompt,
        }
    )


def _answer_fingerprint(
    *, policy_checkpoint_sha: str, decode_config_hash: str, answer_text: str
) -> str:
    """Fingerprint over the *generated output*, independent of the prompt.

    Deliberately excludes the prompt so that the same answer recurring for a
    different prompt (mode collapse onto a generic fallback) is caught as a
    duplicate rather than accepted twice under two different prompt keys.
    """
    return content_sha(
        {
            "family": SOURCE_FAMILY,
            "policy_checkpoint_sha": policy_checkpoint_sha,
            "decode_config_hash": decode_config_hash,
            "answer": answer_text,
        }
    )


def _existing_fingerprints(store: TraceStore) -> tuple[set[str], set[str]]:
    """Scan the store for this collector's prior fingerprints (resume support)."""
    prompt_fingerprints: set[str] = set()
    answer_fingerprints: set[str] = set()
    for row in store.iter_traces():
        meta = row.get("meta") or {}
        if meta.get("source") != SOURCE_FAMILY:
            continue
        prompt_fp = meta.get("prompt_fingerprint")
        answer_fp = meta.get("answer_fingerprint")
        if prompt_fp:
            prompt_fingerprints.add(str(prompt_fp))
        if answer_fp:
            answer_fingerprints.add(str(answer_fp))
    return prompt_fingerprints, answer_fingerprints


def collect_on_policy_traces(
    *,
    generate: GenerateAbstractTrace,
    policy_checkpoint_sha: str,
    prompts: Sequence[str],
    store: TraceStore,
    tokenizer_sha: str = "",
    decode_config_hash: str = "",
    config: SelfDistillCollectConfig | None = None,
    verifier: VerifierCascade | None = None,
) -> CollectionSummary:
    """Collect on-policy abstract-CoT self-distillation traces into ``store``.

    ``generate`` is called once per not-yet-collected, not-yet-duplicate
    prompt in ``prompts`` (in order); its result is verified independently via
    ``verifier`` (defaults to :func:`~slm_training.evals.verifier_cascade.default_openui_cascade`)
    before being labeled accepted/rejected and appended. Both outcomes are
    written -- rejection never means silent deletion.
    """
    cfg = config or SelfDistillCollectConfig()
    candidate_count = len(prompts)
    if not cfg.enabled:
        return CollectionSummary(
            candidate_count=candidate_count,
            shard_count=0,
            processed=0,
            skipped_resumed=0,
            duplicate=0,
            accepted=0,
            rejected=0,
        )

    shard = tuple(prompts[cfg.shard_index :: cfg.num_shards])
    cascade = verifier or default_openui_cascade()
    seen_prompt_fps, seen_answer_fps = _existing_fingerprints(store)

    processed = 0
    skipped_resumed = 0
    duplicate = 0
    accepted = 0
    rejected = 0
    trace_length_counts: dict[int, int] = {}

    for index, prompt in enumerate(shard):
        if cfg.max_records is not None and processed >= cfg.max_records:
            break

        prompt_fp = _prompt_fingerprint(
            policy_checkpoint_sha=policy_checkpoint_sha,
            decode_config_hash=decode_config_hash,
            prompt=prompt,
        )
        if prompt_fp in seen_prompt_fps:
            skipped_resumed += 1
            continue

        generation = generate(prompt)
        capture = generation.capture
        answer_fp = _answer_fingerprint(
            policy_checkpoint_sha=policy_checkpoint_sha,
            decode_config_hash=decode_config_hash,
            answer_text=generation.text,
        )
        # A repeated prompt within this same run must not re-trigger
        # generation either, so mark it seen regardless of outcome.
        seen_prompt_fps.add(prompt_fp)
        if answer_fp in seen_answer_fps:
            duplicate += 1
            continue

        cascade_result = cascade.evaluate(
            candidate_id=f"{SOURCE_FAMILY}:{index}", source=generation.text
        )
        abstract_length = capture.abstract.token_count
        trace_length_counts[abstract_length] = trace_length_counts.get(abstract_length, 0) + 1

        is_accepted = cascade_result.final_status is Verdict.PASS
        reject_reason: str | None = None
        if not is_accepted:
            reject_reason = f"verifier_{cascade_result.final_status.value.lower()}"
        if is_accepted and cfg.reject_forced_end and capture.forced_end:
            is_accepted = False
            reject_reason = "forced_end_cap_rejected_by_config"
        if is_accepted and abstract_length < cfg.min_abstract_tokens:
            is_accepted = False
            reject_reason = "abstract_trace_shorter_than_min_abstract_tokens"

        trace_logprob = sum(
            logprob
            for logprob in (*capture.abstract.logprobs, *capture.answer.logprobs)
            if logprob is not None
        )

        trace = {
            "version": TRACE_VERSION,
            "kind": "decode",
            "meta": {
                "source": SOURCE_FAMILY,
                "prompt": prompt,
                "prompt_fingerprint": prompt_fp,
                "answer_fingerprint": answer_fp,
                "policy_checkpoint_sha": policy_checkpoint_sha,
                "tokenizer_sha": tokenizer_sha,
                "decode_config_hash": decode_config_hash,
                "plan_fingerprint": capture.plan_fingerprint,
                "abstract_termination": capture.abstract_termination,
                "forced_end": capture.forced_end,
                "stop_reason": capture.stop_reason,
                "abstract_token_count": capture.abstract.token_count,
                "answer_token_count": capture.answer.token_count,
                "trace_logprob": trace_logprob,
            },
            "final": {"text": generation.text},
            "labels": {
                "accepted": is_accepted,
                "exact_gold": False,
                "verifier_status": cascade_result.final_status.value,
                "reject_reason": reject_reason,
            },
            "verifier": cascade_result.to_dict(),
            "capture": capture.to_dict(),
        }
        store.append(trace)
        seen_answer_fps.add(answer_fp)
        processed += 1
        if is_accepted:
            accepted += 1
        else:
            rejected += 1

    return CollectionSummary(
        candidate_count=candidate_count,
        shard_count=len(shard),
        processed=processed,
        skipped_resumed=skipped_resumed,
        duplicate=duplicate,
        accepted=accepted,
        rejected=rejected,
        trace_length_counts=trace_length_counts,
    )


def segment_ids_for_capture(capture: AbstractTraceCapture) -> Any:
    """AP-019 ``SegmentKind`` ids (``[1, seq]``) for one captured trace.

    Order is prompt -> abstract -> target, with no privileged-plan segment.
    """
    import torch

    from slm_training.models.block_attention import SegmentKind

    ids = (
        [int(SegmentKind.PROMPT)] * capture.prompt.token_count
        + [int(SegmentKind.ABSTRACT)] * capture.abstract.token_count
        + [int(SegmentKind.TARGET)] * capture.answer.token_count
    )
    return torch.tensor([ids], dtype=torch.long)


def training_example_from_capture(capture: AbstractTraceCapture) -> dict[str, Any]:
    """Build the ``forward_with_segments(input_ids, segment_ids)`` call-site inputs.

    Concatenates the captured prompt/abstract/answer token ids and pairs them
    with :func:`segment_ids_for_capture` so the abstract + target masked SFT
    loss (AP-019) trains directly on a collected on-policy trace.
    """
    import torch

    token_ids = (
        capture.prompt.token_ids + capture.abstract.token_ids + capture.answer.token_ids
    )
    return {
        "input_ids": torch.tensor([list(token_ids)], dtype=torch.long),
        "segment_ids": segment_ids_for_capture(capture),
    }
