"""Codebook-only abstract trace decoding with a hard cap and forced end (AP-017 / SLM-304).

An opt-in, default-off decode path that generates an **abstract span** using only
the reserved AbstractPlanV1 codebook (AP-016) plus ``<endabstract>``, then hands
control to the existing constrained program decoder for the answer phase:

* ``prompt``   — the caller-supplied prefix; no tokens are generated in this phase.
* ``abstract`` — every forward's logits are hard-masked to the M codebook slot ids
  plus ``<endabstract>``; no out-of-codebook token can be selected. The phase ends
  when ``<endabstract>`` is sampled or when ``m_max`` slot tokens have been emitted,
  in which case the end delimiter is **forced** (appended without a forward, like
  any singleton deterministic commit) and recorded as ``forced_end``.
* ``answer``   — the existing program decoder discipline (singleton bypass without
  a forward, legal-set-masked selection) continues from the extended prefix.

Fail closed: a malformed or nested abstract delimiter — in the incoming prompt,
in a hand-checked span, or emitted by the answer phase — raises
:class:`AbstractDecodeError`; there is no repair path and no full-vocabulary
fallback. Abstract and answer sampling carry independent temperature/seed pairs;
``temperature == 0.0`` is greedy (deterministic). Per-phase token counts and
per-token log probabilities are recorded for both generated phases; a token
committed without a forward (the phase-opening ``<beginabstract>``, a forced cap
end, or a singleton bypass) records ``None``.

Like :mod:`slm_training.models.causal_trace`, the core is torch-free: the model
is an injected ``forward_logits(prefix_ids)`` callable and the answer legality is
an injected ``allowed_ids`` callable, so the whole path is deterministic and
unit-testable with fixture logits.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from slm_training.dsl.abstract_plan import AbstractPlanV1
from slm_training.models.causal_trace import AllowedIds, ForwardLogits

__all__ = [
    "AbstractDecodeConfig",
    "AbstractDecodeError",
    "AbstractPhaseRecord",
    "AbstractSamplingParams",
    "AbstractTraceCapture",
    "AbstractTracedGeneration",
    "DecodePhase",
    "abstract_legal_token_ids",
    "capture_abstract_trace",
    "mask_logits_to_abstract_codebook",
    "validate_abstract_delimiters",
]


class AbstractDecodeError(ValueError):
    """Raised when an abstract trace is malformed (fail closed, never repaired)."""


class DecodePhase(str, Enum):
    """The three phases of an abstract-trace decode."""

    PROMPT = "prompt"
    ABSTRACT = "abstract"
    ANSWER = "answer"


@dataclass(frozen=True)
class AbstractSamplingParams:
    """Independent sampling parameters for one phase (``temperature == 0`` is greedy)."""

    temperature: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "seed", int(self.seed))
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")


@dataclass(frozen=True)
class AbstractDecodeConfig:
    """Default-off config for codebook-only abstract trace decoding.

    ``CausalLMOpenUIConfig.abstract`` is ``None`` unless this is explicitly
    supplied; ``None`` keeps every legacy path byte-identical. ``m_max`` is the
    hard cap on emitted abstract slot tokens and defaults to the plan's own
    ``max_slot_count``.
    """

    plan: AbstractPlanV1 = field(default_factory=AbstractPlanV1)
    m_max: int | None = None
    abstract: AbstractSamplingParams = field(default_factory=AbstractSamplingParams)
    answer: AbstractSamplingParams = field(default_factory=AbstractSamplingParams)

    def __post_init__(self) -> None:
        if self.m_max is not None:
            if int(self.m_max) < 1:
                raise ValueError("m_max must be >= 1")
            if int(self.m_max) > self.plan.max_slot_count:
                raise ValueError(
                    f"m_max {self.m_max} exceeds the plan's max_slot_count "
                    f"({self.plan.max_slot_count})"
                )
            object.__setattr__(self, "m_max", int(self.m_max))

    @property
    def effective_m_max(self) -> int:
        return self.m_max if self.m_max is not None else self.plan.max_slot_count


@dataclass(frozen=True)
class AbstractPhaseRecord:
    """Tokens emitted in one phase with their per-token log probabilities.

    A ``None`` logprob marks a token committed without a neural forward: the
    phase-opening ``<beginabstract>`` (a singleton deterministic commit), a
    forced cap end, or a singleton legal-set bypass.
    """

    phase: DecodePhase
    token_ids: tuple[int, ...]
    logprobs: tuple[float | None, ...]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.logprobs):
            raise ValueError("token_ids and logprobs must have equal length")

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "token_ids": list(self.token_ids),
            "logprobs": list(self.logprobs),
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class AbstractTraceCapture:
    """The full three-phase outcome of an abstract-trace decode."""

    plan_fingerprint: str
    prompt: AbstractPhaseRecord
    abstract: AbstractPhaseRecord
    answer: AbstractPhaseRecord
    abstract_termination: str  # "sampled_end" | "forced_cap"
    forced_end: bool
    stop_reason: str  # answer-phase stop: "eos" | "no_legal_continuation" | "max_new_tokens"

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_fingerprint": self.plan_fingerprint,
            "prompt": self.prompt.to_dict(),
            "abstract": self.abstract.to_dict(),
            "answer": self.answer.to_dict(),
            "abstract_termination": self.abstract_termination,
            "forced_end": self.forced_end,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class AbstractTracedGeneration:
    """The result of an abstract-trace generation on the plug-in."""

    text: str
    capture: AbstractTraceCapture
    valid: bool


def abstract_legal_token_ids(plan: AbstractPlanV1) -> tuple[int, ...]:
    """The only token ids legal during the abstract phase: M slots + ``<endabstract>``."""
    return (*plan.slot_token_ids, plan.delimiter_token_ids[1])


def mask_logits_to_abstract_codebook(
    logits: Sequence[float], plan: AbstractPlanV1
) -> list[float]:
    """Hard-mask a logit row to the codebook slots plus ``<endabstract>`` (``-inf`` elsewhere).

    Fails closed when the row is too short to score every codebook id — an
    unscorable legal id is a constrained dead end, never a full-vocabulary
    fallback.
    """
    legal = abstract_legal_token_ids(plan)
    if not logits:
        raise AbstractDecodeError("forward_logits returned an empty logit vector")
    if any(token >= len(logits) for token in legal):
        raise AbstractDecodeError(
            "abstract codebook token id out of logit range (fail closed)"
        )
    masked = [float("-inf")] * len(logits)
    for token in legal:
        masked[token] = float(logits[token])
    return masked


def validate_abstract_delimiters(
    token_ids: Sequence[int], plan: AbstractPlanV1
) -> None:
    """Fail closed unless every abstract delimiter block is well-formed and unnested.

    Well-formed means: zero or more ``<beginabstract> <slot>* <endabstract>``
    blocks; ``<endabstract>`` never appears without a matching open begin; a
    begin inside an open block (nested) or any non-slot token inside a block is
    a hard error. A dangling (unclosed) begin is a hard error.
    """
    begin_id, end_id = plan.delimiter_token_ids
    slots = set(plan.slot_token_ids)
    open_block = False
    for token in token_ids:
        token = int(token)
        if not open_block:
            if token == begin_id:
                open_block = True
            elif token == end_id:
                raise AbstractDecodeError(
                    "malformed abstract trace: <endabstract> without <beginabstract>"
                )
        else:
            if token == begin_id:
                raise AbstractDecodeError(
                    "malformed abstract trace: nested <beginabstract>"
                )
            if token == end_id:
                open_block = False
            elif token not in slots:
                raise AbstractDecodeError(
                    "malformed abstract trace: non-codebook token inside an "
                    "abstract block"
                )
    if open_block:
        raise AbstractDecodeError(
            "malformed abstract trace: unclosed <beginabstract> block"
        )


def _log_softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable log-softmax over a full vocabulary logit row."""
    peak = max(logits)
    shifted = [float(value) - peak for value in logits]
    log_denom = math.log(sum(math.exp(value) for value in shifted))
    return [value - log_denom for value in shifted]


def _select(
    logits: Sequence[float],
    legal: Sequence[int],
    params: AbstractSamplingParams,
    rng: random.Random,
) -> int:
    """Greedy (``temperature == 0``) or seeded temperature sampling over ``legal``."""
    if params.temperature == 0.0:
        return int(max(legal, key=lambda token: (logits[token], -token)))
    peak = max(logits[token] for token in legal)
    weights = [
        math.exp((float(logits[token]) - peak) / params.temperature) for token in legal
    ]
    total = sum(weights)
    draw = rng.random() * total
    cumulative = 0.0
    for token, weight in zip(legal, weights):
        cumulative += weight
        if draw <= cumulative:
            return int(token)
    return int(legal[-1])  # pragma: no cover - float rounding guard


def _forward_checked(
    forward_logits: ForwardLogits, prefix: tuple[int, ...]
) -> tuple[list[float], list[float]]:
    """Run one forward, returning ``(logits, log_probs)``; fail closed on bad rows."""
    logits = [float(value) for value in forward_logits(prefix)]
    if not logits:
        raise AbstractDecodeError("forward_logits returned an empty logit vector")
    if not all(math.isfinite(value) for value in logits):
        raise AbstractDecodeError("forward_logits returned non-finite logits")
    return logits, _log_softmax(logits)


def capture_abstract_trace(
    *,
    forward_logits: ForwardLogits,
    answer_allowed_ids: AllowedIds,
    plan: AbstractPlanV1,
    eos_id: int,
    max_answer_tokens: int,
    initial_prefix: Sequence[int] = (),
    m_max: int | None = None,
    abstract_params: AbstractSamplingParams | None = None,
    answer_params: AbstractSamplingParams | None = None,
) -> AbstractTraceCapture:
    """Decode prompt -> abstract (codebook-only, hard cap) -> answer (program).

    The abstract phase opens with a deterministic ``<beginabstract>`` commit (no
    forward), then each step masks logits to the M codebook slots plus
    ``<endabstract>`` — no out-of-codebook token can be selected. The phase ends
    by sampled end or by forced end at ``m_max`` (recorded as ``forced_end``).
    The answer phase then runs the existing constrained program-decode
    discipline (singleton bypass without a forward; legal-set-masked selection)
    from the extended prefix, and fails closed if it ever emits an abstract
    codebook or delimiter token (a nested/malformed delimiter is a hard error,
    not a repair).
    """
    abstract_params = abstract_params or AbstractSamplingParams()
    answer_params = answer_params or AbstractSamplingParams()
    cap = int(m_max) if m_max is not None else plan.max_slot_count
    if cap < 1 or cap > plan.max_slot_count:
        raise AbstractDecodeError(
            f"m_max must be within [1, {plan.max_slot_count}], got {cap}"
        )
    if int(max_answer_tokens) < 0:
        raise AbstractDecodeError("max_answer_tokens must be non-negative")
    prompt = tuple(int(token) for token in initial_prefix)
    validate_abstract_delimiters(prompt, plan)
    begin_id, end_id = plan.delimiter_token_ids
    codebook = set(plan.token_ids)
    legal_abstract = abstract_legal_token_ids(plan)
    abstract_rng = random.Random(abstract_params.seed)
    answer_rng = random.Random(answer_params.seed)

    # Abstract phase: deterministic <beginabstract> commit (singleton, no forward),
    # then codebook-only steps until sampled or forced end.
    prefix = (*prompt, begin_id)
    abstract_tokens: list[int] = [begin_id]
    abstract_logprobs: list[float | None] = [None]
    forced_end = False
    for _ in range(cap):
        logits, log_probs = _forward_checked(forward_logits, prefix)
        mask_logits_to_abstract_codebook(logits, plan)  # fail closed on range
        selected = _select(logits, legal_abstract, abstract_params, abstract_rng)
        if selected not in legal_abstract:  # defensive: selection can never leave the codebook
            raise AbstractDecodeError("out-of-codebook abstract selection")
        abstract_tokens.append(selected)
        abstract_logprobs.append(float(log_probs[selected]))
        prefix = (*prefix, selected)
        if selected == end_id:
            break
    else:
        forced_end = True
        abstract_tokens.append(end_id)
        abstract_logprobs.append(None)  # forced commit: no forward scored it
        prefix = (*prefix, end_id)
    validate_abstract_delimiters(tuple(abstract_tokens), plan)

    # Answer phase: the existing constrained program decoder discipline.
    answer_tokens: list[int] = []
    answer_logprobs: list[float | None] = []
    stop_reason = "max_new_tokens"
    for _ in range(int(max_answer_tokens)):
        legal = tuple(int(token) for token in answer_allowed_ids(prefix))
        if not legal:
            stop_reason = "no_legal_continuation"
            break
        legal_set = set(legal)
        if len(legal_set) == 1:
            selected = next(iter(legal_set))
            answer_tokens.append(int(selected))
            answer_logprobs.append(None)  # singleton bypass: no forward
            prefix = (*prefix, int(selected))
            if selected == eos_id:
                stop_reason = "eos"
                break
            continue
        logits, log_probs = _forward_checked(forward_logits, prefix)
        if any(token < 0 or token >= len(logits) for token in legal_set):
            raise AbstractDecodeError("legal token id out of logit range")
        selected = _select(logits, tuple(sorted(legal_set)), answer_params, answer_rng)
        if selected in codebook:
            raise AbstractDecodeError(
                "malformed abstract trace: answer phase emitted an abstract "
                "codebook/delimiter token (nested delimiters are a hard error)"
            )
        answer_tokens.append(int(selected))
        answer_logprobs.append(float(log_probs[selected]))
        prefix = (*prefix, int(selected))
        if selected == eos_id:
            stop_reason = "eos"
            break
    if any(token in codebook for token in answer_tokens):
        raise AbstractDecodeError(
            "malformed abstract trace: answer phase emitted an abstract "
            "codebook/delimiter token (nested delimiters are a hard error)"
        )

    return AbstractTraceCapture(
        plan_fingerprint=plan.compatibility_fingerprint,
        prompt=AbstractPhaseRecord(DecodePhase.PROMPT, prompt, tuple(None for _ in prompt)),
        abstract=AbstractPhaseRecord(
            DecodePhase.ABSTRACT, tuple(abstract_tokens), tuple(abstract_logprobs)
        ),
        answer=AbstractPhaseRecord(
            DecodePhase.ANSWER, tuple(answer_tokens), tuple(answer_logprobs)
        ),
        abstract_termination=("forced_cap" if forced_end else "sampled_end"),
        forced_end=forced_end,
        stop_reason=stop_reason,
    )
