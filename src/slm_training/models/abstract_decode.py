"""Codebook-only abstract-trace decoding with a hard cap and forced end (AP-017 / SLM-304).

AP-016 (:mod:`slm_training.dsl.abstract_plan`) reserved a collision-free
``<beginabstract>`` / codebook-slot / ``<endabstract>`` vocabulary but never decoded
through it. This module adds the torch-free decode core for the ``abstract`` phase: once
``<beginabstract>`` has been committed, the legal set at every step is masked to *exactly*
the plan's ``M`` codebook tokens plus ``<endabstract>`` -- never the full vocabulary and
never a token outside the reserved namespace. If the model never selects
``<endabstract>`` within ``max_slot_count`` codebook tokens, the legal set collapses to
``{<endabstract>}`` alone and :func:`slm_training.models.causal_trace.capture_raw_steps`'s
existing singleton-forced fast path deterministically emits it -- the hard cap and forced
end are the same "sole legal continuation" mechanism the DSL grammar path already relies
on, not a new decode primitive.

This module is opt-in and additive: nothing in the repository calls it yet, so it cannot
change any existing decode output. A caller hands control to the ordinary DSL-grammar
``answer``/``program`` phase once :func:`capture_abstract_span` returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from slm_training.dsl.abstract_plan import AbstractPlanV1
from slm_training.models.causal_trace import (
    AllowedIds,
    CaptureResult,
    ForwardLogits,
    TracePolicy,
    TraceSelection,
    capture_raw_steps,
)

__all__ = [
    "AbstractDecodeError",
    "AbstractSpanCapture",
    "PhasedGenerationCapture",
    "capture_abstract_span",
    "capture_phased_generation",
    "validate_abstract_span",
]


class AbstractDecodeError(ValueError):
    """Raised when an abstract span violates the AP-017 codebook-only contract."""


@dataclass(frozen=True)
class AbstractSpanCapture:
    """One captured ``<beginabstract>...<endabstract>`` span: codebook-only, hard-capped.

    ``slot_token_ids`` excludes the begin/end delimiters. ``forced_end`` is ``True`` only
    when the span reached ``plan.max_slot_count`` codebook tokens and ``<endabstract>``
    was the sole legal continuation (a deterministic deduction), and ``False`` when the
    model selected ``<endabstract>`` as an ordinary decision before the cap. ``log_probs``
    holds one entry per emitted token (codebook slots and the trailing end token, in
    order); a forced token's entry is ``0.0`` (``log(1.0)``) because it had no alternative.
    """

    slot_token_ids: tuple[int, ...]
    forced_end: bool
    token_count: int
    log_probs: tuple[float, ...]
    capture: CaptureResult

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_token_ids": list(self.slot_token_ids),
            "forced_end": self.forced_end,
            "token_count": self.token_count,
            "log_probs": list(self.log_probs),
            "stop_reason": self.capture.stop_reason,
        }


def validate_abstract_span(
    token_ids: Sequence[int], plan: AbstractPlanV1
) -> tuple[int, ...]:
    """Fail closed unless ``token_ids`` is exactly one well-formed abstract span.

    A well-formed span is ``<beginabstract>``, followed by zero or more of the plan's
    reserved codebook slot ids and nothing else, followed by exactly one trailing
    ``<endabstract>``. Returns the slot ids (delimiters stripped) on success; raises
    :class:`AbstractDecodeError` on a missing/duplicated delimiter, a nested
    ``<beginabstract>``/``<endabstract>``, an out-of-codebook token, or a body longer
    than ``plan.max_slot_count``.
    """
    begin_id, end_id = plan.delimiter_token_ids
    ids = tuple(int(token) for token in token_ids)
    if len(ids) < 2:
        raise AbstractDecodeError(
            "abstract span must contain at least <beginabstract> and <endabstract>"
        )
    if ids[0] != begin_id:
        raise AbstractDecodeError("abstract span must start with <beginabstract>")
    if ids[-1] != end_id:
        raise AbstractDecodeError("abstract span must end with <endabstract>")
    body = ids[1:-1]
    if begin_id in body or end_id in body:
        raise AbstractDecodeError(
            "nested <beginabstract>/<endabstract> delimiter is malformed"
        )
    codebook_ids = set(plan.slot_token_ids)
    for token in body:
        if token not in codebook_ids:
            raise AbstractDecodeError(
                f"token {token} is outside the reserved abstract codebook"
            )
    if len(body) > plan.max_slot_count:
        raise AbstractDecodeError(
            f"abstract span has {len(body)} codebook tokens, exceeding the hard cap "
            f"{plan.max_slot_count}"
        )
    return body


def _abstract_allowed_ids(
    plan: AbstractPlanV1, *, start_len: int, end_id: int
) -> AllowedIds:
    """Mask to exactly the plan's codebook + ``<endabstract>``, forcing end at the cap."""
    codebook_ids = tuple(sorted(plan.slot_token_ids))
    max_slots = plan.max_slot_count

    def allowed(prefix: tuple[int, ...]) -> tuple[int, ...]:
        emitted = len(prefix) - start_len
        if emitted < 0:
            raise AbstractDecodeError(
                "abstract decode prefix shrank below its starting length"
            )
        if emitted >= max_slots:
            return (end_id,)
        return (*codebook_ids, end_id)

    return allowed


def capture_abstract_span(
    *,
    forward_logits: ForwardLogits,
    plan: AbstractPlanV1,
    initial_prefix: Sequence[int],
    uniform_seed: int = 0,
) -> AbstractSpanCapture:
    """Decode one codebook-only abstract span, forcing ``<endabstract>`` at the hard cap.

    ``initial_prefix`` must already end with ``<beginabstract>`` -- the caller commits the
    begin delimiter before entering the abstract phase (mirroring the existing
    prompt -> answer phase hand-off in
    :meth:`slm_training.models.causal_lm_openui.CausalLMOpenUIPlugin.generate_constrained_traced`).
    At every step the legal set is masked to exactly ``plan.slot_token_ids`` plus
    ``<endabstract>``, so no out-of-codebook token can ever be selected; once
    ``plan.max_slot_count`` codebook tokens have been emitted, the legal set collapses to
    ``<endabstract>`` alone and is force-emitted deterministically.
    """
    begin_id, end_id = plan.delimiter_token_ids
    prefix = tuple(int(token) for token in initial_prefix)
    if not prefix or prefix[-1] != begin_id:
        raise AbstractDecodeError(
            "capture_abstract_span requires initial_prefix to already end with "
            "<beginabstract>"
        )
    start_len = len(prefix)
    allowed_ids = _abstract_allowed_ids(plan, start_len=start_len, end_id=end_id)
    capture = capture_raw_steps(
        forward_logits=forward_logits,
        allowed_ids=allowed_ids,
        eos_id=end_id,
        max_new_tokens=plan.max_slot_count + 1,
        initial_prefix=prefix,
        policy=TracePolicy(selection=TraceSelection.EVERY, top_k=1),
        uniform_seed=uniform_seed,
    )
    if capture.stop_reason != "eos":
        # allowed_ids never returns an empty set and the +1 budget always covers the
        # forced-end step, so only an internal contract violation reaches this branch.
        raise AbstractDecodeError(
            "abstract span decode did not terminate with <endabstract> "
            f"(stop_reason={capture.stop_reason!r})"
        )
    body = validate_abstract_span((begin_id, *capture.generated_token_ids), plan)
    forced_end = len(body) >= plan.max_slot_count
    log_probs = _phase_log_probs(capture)
    return AbstractSpanCapture(
        slot_token_ids=body,
        forced_end=forced_end,
        token_count=len(capture.generated_token_ids),
        log_probs=log_probs,
        capture=capture,
    )


def _phase_log_probs(capture: CaptureResult) -> tuple[float, ...]:
    """Per-emitted-token selected log-prob; forced (singleton-bypass) steps are ``0.0``.

    A forced step has no :class:`~slm_training.models.causal_trace.RawStepObservation`
    (``capture_raw_steps`` never calls the model when the legal set has one member), so
    its selection was certain -- ``log(1.0) == 0.0`` is the honest probability, not a
    placeholder.
    """
    by_ordinal = {obs.generated_ordinal: obs for obs in capture.observations}
    log_probs: list[float] = []
    for ordinal in range(len(capture.generated_token_ids)):
        obs = by_ordinal.get(ordinal)
        if obs is None:
            log_probs.append(0.0)
            continue
        prob = next(p for token, p in obs.legal_topk if token == obs.selected_token_id)
        log_probs.append(math.log(prob))
    return tuple(log_probs)


@dataclass(frozen=True)
class PhasedGenerationCapture:
    """``prompt`` -> ``abstract`` -> ``answer``/``program`` decode, phase-tagged.

    ``prompt_token_count`` is the length of the caller-supplied prompt prefix (no
    decoding happens in that phase); ``abstract`` is the full codebook-only span
    capture; ``answer`` is the ordinary (grammar-constrained or otherwise)
    continuation capture that starts immediately after ``<endabstract>``.
    """

    prompt_token_count: int
    abstract: AbstractSpanCapture
    answer: CaptureResult

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_token_count": self.prompt_token_count,
            "abstract": self.abstract.to_dict(),
            "answer": self.answer.to_dict(),
        }


def capture_phased_generation(
    *,
    forward_logits: ForwardLogits,
    plan: AbstractPlanV1,
    prompt_ids: Sequence[int],
    answer_allowed_ids: AllowedIds,
    answer_eos_id: int,
    answer_max_new_tokens: int,
    abstract_seed: int = 0,
    answer_seed: int = 0,
) -> PhasedGenerationCapture:
    """Decode the ``prompt`` / ``abstract`` / ``answer`` phases with independent seeds.

    ``<beginabstract>`` is appended to ``prompt_ids`` to open the abstract phase; once it
    closes with ``<endabstract>`` (sampled or forced), ``answer_allowed_ids`` takes over
    for an ordinary constrained decode (e.g. the existing DSL-grammar mask) -- the two
    phases never share a sampling seed, so re-running one independently reproduces only
    that phase.
    """
    begin_id, _end_id = plan.delimiter_token_ids
    prompt = tuple(int(token) for token in prompt_ids)
    abstract = capture_abstract_span(
        forward_logits=forward_logits,
        plan=plan,
        initial_prefix=(*prompt, begin_id),
        uniform_seed=abstract_seed,
    )
    answer_prefix = (*prompt, begin_id, *abstract.capture.generated_token_ids)
    answer = capture_raw_steps(
        forward_logits=forward_logits,
        allowed_ids=answer_allowed_ids,
        eos_id=answer_eos_id,
        max_new_tokens=answer_max_new_tokens,
        initial_prefix=answer_prefix,
        uniform_seed=answer_seed,
    )
    return PhasedGenerationCapture(
        prompt_token_count=len(prompt), abstract=abstract, answer=answer
    )
