"""Tests for codebook-only abstract trace decoding (AP-017 / SLM-304).

The torch-free core is exercised with fixture logits over a small AbstractPlanV1
codebook (M=4, m_max=8); torch-gated plug-in tests drive the real per-step
forward through a tiny deterministic model. Covered: early end, exact cap
(forced termination recorded), empty trace, malformed/nested delimiter
rejection, no out-of-codebook token, deterministic seeded sampling, per-phase
counts/logprobs, and default-off parity of the legacy paths.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.abstract_plan import AbstractPlanV1
from slm_training.models.abstract_decode import (
    AbstractDecodeConfig,
    AbstractDecodeError,
    AbstractSamplingParams,
    DecodePhase,
    abstract_legal_token_ids,
    capture_abstract_trace,
    mask_logits_to_abstract_codebook,
    validate_abstract_delimiters,
)

PLAN = AbstractPlanV1(slot_count=4, max_slot_count=8)
BEGIN_ID, END_ID = PLAN.delimiter_token_ids
SLOT_IDS = PLAN.slot_token_ids
VOCAB = max(PLAN.token_ids) + 1
EOS_ID = 0


def _row(winner: int, *, runner_up: int | None = None) -> list[float]:
    logits = [0.01 * (i % 7) for i in range(VOCAB)]
    if runner_up is not None:
        logits[runner_up] = 2.0
    logits[winner] = 5.0
    return logits


def _forward(winners: dict[tuple[int, ...], int]):
    def forward(prefix: tuple[int, ...]) -> list[float]:
        return _row(winners.get(tuple(prefix), END_ID))

    return forward


def _answer_legal(steps: list[tuple[int, ...]]):
    """Legal sets for the answer phase, consumed in order by suffix length."""

    def allowed(prefix: tuple[int, ...]) -> tuple[int, ...]:
        # The abstract block occupies len(abstract) ids after the prompt; the
        # answer suffix is whatever came after the last delimiter.
        suffix = len(prefix) - prefix.index(END_ID) - 1 if END_ID in prefix else 0
        return steps[min(suffix, len(steps) - 1)]

    return allowed


def _capture(winners, *, m_max=None, abstract_params=None, answer_params=None,
             answer_steps=((5, 6), (EOS_ID,))):
    return capture_abstract_trace(
        forward_logits=_forward(winners),
        answer_allowed_ids=_answer_legal(list(answer_steps)),
        plan=PLAN,
        eos_id=EOS_ID,
        max_answer_tokens=8,
        initial_prefix=(1,),
        m_max=m_max,
        abstract_params=abstract_params,
        answer_params=answer_params,
    )


def test_early_sampled_end() -> None:
    capture = _capture({(1, BEGIN_ID): END_ID})
    assert capture.abstract.token_ids == (BEGIN_ID, END_ID)
    assert capture.abstract_termination == "sampled_end"
    assert capture.forced_end is False
    assert capture.stop_reason == "eos"


def test_exact_cap_forces_end_and_records_it() -> None:
    winners = {
        (1, BEGIN_ID): SLOT_IDS[0],
        (1, BEGIN_ID, SLOT_IDS[0]): SLOT_IDS[1],
        (1, BEGIN_ID, SLOT_IDS[0], SLOT_IDS[1]): SLOT_IDS[2],
    }
    capture = _capture(winners, m_max=3)
    assert capture.abstract.token_ids == (
        BEGIN_ID, SLOT_IDS[0], SLOT_IDS[1], SLOT_IDS[2], END_ID,
    )
    assert capture.forced_end is True
    assert capture.abstract_termination == "forced_cap"
    # The forced end is committed without a forward: its logprob is None.
    assert capture.abstract.logprobs[-1] is None
    # The deterministic <beginabstract> commit is likewise forward-free.
    assert capture.abstract.logprobs[0] is None
    assert all(
        lp is None or lp < 0.0 for lp in capture.abstract.logprobs
    )


def test_empty_abstract_span() -> None:
    capture = _capture({(1, BEGIN_ID): END_ID})
    slots = [t for t in capture.abstract.token_ids if t in SLOT_IDS]
    assert slots == []
    assert capture.forced_end is False


def test_no_out_of_codebook_token_even_when_raw_argmax_is_outside() -> None:
    # The raw argmax (id 7) is outside the codebook; masking forbids selecting it.
    capture = _capture(
        {(1, BEGIN_ID): 7, (1, BEGIN_ID, END_ID): 7},
        answer_steps=((EOS_ID,),),
    )
    # With id 7 masked out the best codebook row wins; every abstract token is
    # a delimiter or a codebook slot.
    assert set(capture.abstract.token_ids) <= set(PLAN.token_ids)
    assert capture.abstract.token_ids[0] == BEGIN_ID
    assert capture.abstract.token_ids[-1] == END_ID


def test_mask_logits_fails_closed_on_short_row() -> None:
    with pytest.raises(AbstractDecodeError, match="out of logit range"):
        mask_logits_to_abstract_codebook([0.0] * 8, PLAN)
    masked = mask_logits_to_abstract_codebook(_row(7), PLAN)
    legal = set(abstract_legal_token_ids(PLAN))
    for token, value in enumerate(masked):
        assert (value > float("-inf")) == (token in legal)


def test_malformed_delimiters_are_hard_errors() -> None:
    with pytest.raises(AbstractDecodeError, match="without <beginabstract>"):
        validate_abstract_delimiters((END_ID,), PLAN)
    with pytest.raises(AbstractDecodeError, match="nested"):
        validate_abstract_delimiters((BEGIN_ID, SLOT_IDS[0], BEGIN_ID, END_ID), PLAN)
    with pytest.raises(AbstractDecodeError, match="non-codebook"):
        validate_abstract_delimiters((BEGIN_ID, 7, END_ID), PLAN)
    with pytest.raises(AbstractDecodeError, match="unclosed"):
        validate_abstract_delimiters((BEGIN_ID, SLOT_IDS[0]), PLAN)
    validate_abstract_delimiters((BEGIN_ID, SLOT_IDS[0], END_ID, 7), PLAN)  # ok


def test_malformed_prompt_prefix_is_rejected() -> None:
    with pytest.raises(AbstractDecodeError, match="without <beginabstract>"):
        capture_abstract_trace(
            forward_logits=_forward({}),
            answer_allowed_ids=lambda _p: (EOS_ID,),
            plan=PLAN,
            eos_id=EOS_ID,
            max_answer_tokens=4,
            initial_prefix=(1, END_ID),
        )


def test_answer_phase_rejects_nested_delimiters() -> None:
    def nested_allowed(prefix: tuple[int, ...]) -> tuple[int, ...]:
        return (SLOT_IDS[0], EOS_ID)  # a codebook token offered to the answer phase

    with pytest.raises(AbstractDecodeError, match="answer phase emitted"):
        capture_abstract_trace(
            forward_logits=_forward({(1, BEGIN_ID): END_ID}),
            answer_allowed_ids=nested_allowed,
            plan=PLAN,
            eos_id=EOS_ID,
            max_answer_tokens=4,
            initial_prefix=(1,),
        )


def test_seeded_sampling_is_deterministic_and_independent() -> None:
    winners = {}
    params = AbstractSamplingParams(temperature=1.0, seed=7)
    first = _capture(winners, m_max=8, abstract_params=params)
    second = _capture(winners, m_max=8, abstract_params=params)
    assert first == second
    # Greedy decoding ignores the seed and always takes the masked argmax.
    greedy_a = _capture(winners, m_max=8,
                        abstract_params=AbstractSamplingParams(seed=1))
    greedy_b = _capture(winners, m_max=8,
                        abstract_params=AbstractSamplingParams(seed=999))
    assert greedy_a.abstract == greedy_b.abstract
    # Independent seeds: changing only the answer seed leaves the abstract span alone.
    answer_a = _capture(winners, answer_params=AbstractSamplingParams(1.0, 3))
    answer_b = _capture(winners, answer_params=AbstractSamplingParams(1.0, 4))
    assert answer_a.abstract == answer_b.abstract


def test_per_phase_counts_and_logprobs() -> None:
    capture = _capture({(1, BEGIN_ID): SLOT_IDS[2],
                        (1, BEGIN_ID, SLOT_IDS[2]): END_ID})
    assert capture.prompt.phase is DecodePhase.PROMPT
    assert capture.prompt.token_count == 1  # the initial prefix, ungenerated
    assert capture.abstract.phase is DecodePhase.ABSTRACT
    assert capture.abstract.token_count == 3  # begin, slot, end
    assert capture.answer.phase is DecodePhase.ANSWER
    for record in (capture.abstract, capture.answer):
        assert len(record.logprobs) == record.token_count
    # Sampled tokens carry finite log probabilities; bypassed ones carry None.
    assert capture.abstract.logprobs[1] is not None
    assert capture.answer.logprobs[-1] is None  # singleton EOS bypass
    payload = capture.to_dict()
    assert payload["abstract"]["token_count"] == 3
    assert payload["forced_end"] is False
    assert payload["plan_fingerprint"] == PLAN.compatibility_fingerprint


def test_answer_phase_stop_reasons() -> None:
    capture = _capture({}, answer_steps=((5, 6),))  # never reaches EOS
    assert capture.stop_reason == "max_new_tokens"
    empty = capture_abstract_trace(
        forward_logits=_forward({}),
        answer_allowed_ids=lambda _p: (),
        plan=PLAN,
        eos_id=EOS_ID,
        max_answer_tokens=4,
        initial_prefix=(1,),
    )
    assert empty.stop_reason == "no_legal_continuation"
    assert empty.answer.token_count == 0


def test_config_validates_m_max() -> None:
    assert AbstractDecodeConfig(plan=PLAN).effective_m_max == 8
    assert AbstractDecodeConfig(plan=PLAN, m_max=3).effective_m_max == 3
    with pytest.raises(ValueError, match="m_max"):
        AbstractDecodeConfig(plan=PLAN, m_max=0)
    with pytest.raises(ValueError, match="max_slot_count"):
        AbstractDecodeConfig(plan=PLAN, m_max=9)
    with pytest.raises(ValueError, match="temperature"):
        AbstractSamplingParams(temperature=-0.1)


torch = pytest.importorskip("torch")

from slm_training.models.causal_lm_openui import (  # noqa: E402
    CausalLMOpenUIConfig,
    CausalLMOpenUIPlugin,
)


class _Output:
    def __init__(self, logits: "torch.Tensor") -> None:
        self.logits = logits


class _TinyCausalModel(torch.nn.Module):
    def __init__(self, vocab: int = VOCAB, hidden: int = 4) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)
        self.device = torch.device("cpu")
        self.config = torch.nn.Module()

    def forward(self, input_ids: "torch.Tensor") -> _Output:
        return _Output(self.head(self.emb(input_ids)))

    def state_dict(self, *args, **kwargs):  # keep compatibility_fingerprint cheap
        return {}


class _TinyTokenizer:
    eos_token_id = EOS_ID
    pad_token_id = EOS_ID
    init_kwargs: dict = {}

    def __len__(self) -> int:
        return VOCAB

    def __call__(self, text: str, return_tensors: str = "pt"):
        return {"input_ids": torch.tensor([[1]])}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(str(int(i)) for i in ids)


def _plugin(abstract: AbstractDecodeConfig | None = None) -> CausalLMOpenUIPlugin:
    torch.manual_seed(0)
    config = CausalLMOpenUIConfig(
        base_model_id="tiny", base_model_revision="rev", abstract=abstract
    )
    return CausalLMOpenUIPlugin(_TinyCausalModel(), _TinyTokenizer(), config)


def _eos_only(_prefix: tuple[int, ...]) -> tuple[int, ...]:
    return (EOS_ID,)


def test_plugin_abstract_trace_decode() -> None:
    plugin = _plugin(AbstractDecodeConfig(plan=PLAN, m_max=4))
    out = plugin.generate_abstract_traced("Make a card", allowed_ids_fn=_eos_only)
    capture = out.capture
    assert capture.prompt.token_ids == (1,)
    assert capture.abstract.token_ids[0] == BEGIN_ID
    assert capture.abstract.token_ids[-1] == END_ID
    assert set(capture.abstract.token_ids) <= set(PLAN.token_ids)
    # m_max=4 bounds the span; termination is always sampled or forced end.
    assert len([t for t in capture.abstract.token_ids if t in SLOT_IDS]) <= 4
    assert capture.abstract_termination in {"sampled_end", "forced_cap"}
    assert capture.answer.token_ids == (EOS_ID,)
    assert capture.stop_reason == "eos"
    # The answer text (just EOS) certifies via the honest fallback.
    assert out.text == "root = Separator()"
    assert out.valid is True


def test_plugin_abstract_decode_is_default_off() -> None:
    plugin = _plugin()
    with pytest.raises(ValueError, match="default-off"):
        plugin.generate_abstract_traced("Make a card", allowed_ids_fn=_eos_only)


def test_off_mode_parity_of_legacy_paths(monkeypatch) -> None:
    # Setting (but not using) the abstract config must not change legacy outputs.
    off = _plugin()
    on = _plugin(AbstractDecodeConfig(plan=PLAN, m_max=4))
    legal_by_suffix = {0: (2, 3), 1: (4,), 2: (2, 5), 3: (EOS_ID,)}

    def allowed(prefix: tuple[int, ...]) -> tuple[int, ...]:
        return legal_by_suffix.get(len(prefix) - 1, (EOS_ID,))

    traced_off = off.generate_constrained_traced(
        "Make a card", group_id="grp", allowed_ids_fn=allowed
    )
    traced_on = on.generate_constrained_traced(
        "Make a card", group_id="grp", allowed_ids_fn=allowed
    )
    assert traced_off.result.generated_token_ids == traced_on.result.generated_token_ids
    assert traced_off.text == traced_on.text
    # The untraced legacy path is identical too (grammar seam stubbed, as in the
    # existing plug-in tests, to avoid the full-vocabulary grammar scan).
    monkeypatch.setattr(off, "_allowed_ids", lambda _prefix: (EOS_ID,))
    monkeypatch.setattr(on, "_allowed_ids", lambda _prefix: (EOS_ID,))
    assert off.generate_constrained("Make a card") == on.generate_constrained(
        "Make a card"
    )
