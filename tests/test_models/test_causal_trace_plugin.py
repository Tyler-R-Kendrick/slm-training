"""Torch-gated tests for the causal plug-in traced decode (LDI1-01 / SLM-119).

A tiny deterministic causal model drives the real per-step torch forward through the
capture loop. The grammar legal-set seam is injected so the loop is exercised without
the OpenUI parser. These tests assert the torch wiring, exact logit replay, reproducible
forced-action replay, and that tracing never changes which token is emitted.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.distill.trace_store import TraceStore  # noqa: E402
from slm_training.models.causal_lm_openui import (  # noqa: E402
    CausalLMOpenUIConfig,
    CausalLMOpenUIPlugin,
)
from slm_training.models.causal_trace import (  # noqa: E402
    CausalTraceWriter,
    load_causal_decision_states,
)

# Legality by generated-suffix length (prompt is a single token, so prompt_len == 1).
_LEGAL_BY_SUFFIX = {0: (2, 3), 1: (4,), 2: (0,)}


class _Output:
    def __init__(self, logits: "torch.Tensor") -> None:
        self.logits = logits


class _TinyCausalModel(torch.nn.Module):
    def __init__(self, vocab: int = 6, hidden: int = 4) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)
        self.device = torch.device("cpu")
        self.config = torch.nn.Module()  # a plain attr-bag for decode_config_hash

    def forward(self, input_ids: "torch.Tensor") -> _Output:
        return _Output(self.head(self.emb(input_ids)))

    def state_dict(self, *args, **kwargs):  # keep compatibility_fingerprint cheap
        return {}


class _TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    init_kwargs: dict = {}

    def __len__(self) -> int:
        return 6

    def __call__(self, text: str, return_tensors: str = "pt"):
        return {"input_ids": torch.tensor([[1]])}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(str(int(i)) for i in ids)


def _plugin() -> CausalLMOpenUIPlugin:
    torch.manual_seed(0)
    config = CausalLMOpenUIConfig(base_model_id="tiny", base_model_revision="rev")
    return CausalLMOpenUIPlugin(_TinyCausalModel(), _TinyTokenizer(), config)


def _allowed(prefix: tuple[int, ...]) -> tuple[int, ...]:
    return _LEGAL_BY_SUFFIX[len(prefix) - 1]  # prompt_len == 1


def test_traced_decode_records_states_and_stops_on_eos(tmp_path) -> None:
    plugin = _plugin()
    identity = plugin.capture_identity(group_id="grp", context_text="root=Stack([")
    store = TraceStore(tmp_path / "traces", run_id="ldi1-plugin")
    writer = CausalTraceWriter(store, identity)

    out = plugin.generate_constrained_traced(
        "Make a card", group_id="grp", trace_writer=writer, allowed_ids_fn=_allowed
    )
    assert out.result.generated_token_ids[-1] == 0  # EOS
    assert out.result.stop_reason == "eos"
    # Every emitted token was legal under the injected grammar seam.
    for obs in out.result.observations:
        assert obs.selected_token_id in _allowed(obs.prefix_token_ids)

    states = load_causal_decision_states(
        store,
        expected_checkpoint_sha=identity.policy_checkpoint_sha,
        expected_tokenizer_sha=identity.tokenizer_sha,
    )
    assert len(states) == len(out.result.observations)
    # context_ids are the FULL prefix (prompt + suffix), starting at the prompt token.
    assert states[0].context_ids == (1,)


def test_stored_logits_replay_within_tolerance(tmp_path) -> None:
    plugin = _plugin()
    out = plugin.generate_constrained_traced(
        "Make a card", group_id="grp", allowed_ids_fn=_allowed
    )
    for obs in out.result.observations:
        row = torch.tensor([list(obs.prefix_token_ids)])
        with torch.inference_mode():
            recomputed = plugin.model(row).logits[0, -1, :]
        stored_id, stored_logit, _lp = obs.raw_topk[0]
        assert obs.raw_argmax_id == int(recomputed.argmax(-1))
        assert abs(float(recomputed[stored_id]) - stored_logit) < 1e-4


def test_tracing_does_not_change_emitted_tokens() -> None:
    plugin = _plugin()
    out = plugin.generate_constrained_traced(
        "Make a card", group_id="grp", allowed_ids_fn=_allowed
    )
    # Independently reproduce greedy-over-legal selection from the same model.
    prefix: tuple[int, ...] = (1,)
    expected: list[int] = []
    for _ in range(len(out.result.generated_token_ids)):
        with torch.inference_mode():
            logits = plugin.model(torch.tensor([list(prefix)])).logits[0, -1, :]
        legal = _allowed(prefix)
        chosen = max(legal, key=lambda t: (float(logits[t]), -t))
        expected.append(chosen)
        prefix = (*prefix, chosen)
        if chosen == 0:
            break
    assert list(out.result.generated_token_ids) == expected


def test_forced_action_replay_is_reproducible(tmp_path) -> None:
    plugin = _plugin()
    identity = plugin.capture_identity(group_id="grp", context_text="root=Stack([")
    store = TraceStore(tmp_path / "traces", run_id="ldi1-plugin")
    writer = CausalTraceWriter(store, identity)
    plugin.generate_constrained_traced(
        "Make a card", group_id="grp", trace_writer=writer, allowed_ids_fn=_allowed
    )
    state = load_causal_decision_states(
        store,
        expected_checkpoint_sha=identity.policy_checkpoint_sha,
        expected_tokenizer_sha=identity.tokenizer_sha,
    )[0]

    first = plugin.replay_causal_action(state, 3, 7, allowed_ids_fn=_allowed)
    second = plugin.replay_causal_action(state, 3, 7, allowed_ids_fn=_allowed)
    assert first == second  # deterministic continuation for the same seed
    assert first.action_id == 3  # forced action applied to the exact prefix
    assert first.continuation_seed == 7


def test_active_adapter_identity_reflects_lora_params() -> None:
    plugin = _plugin()
    assert plugin.active_adapter_identity() == ""  # no adapter params on the base model
    base_identity = plugin.capture_identity(group_id="grp", context_text="ctx")

    real = plugin.model.named_parameters
    plugin.model.named_parameters = lambda: [  # type: ignore[assignment]
        ("base_model.layers.0.lora_A.weight", torch.zeros(1)),
        *list(real()),
    ]
    assert plugin.active_adapter_identity() != ""
    adapter_identity = plugin.capture_identity(group_id="grp", context_text="ctx")
    # Adapter presence changes the folded policy fingerprint (hence state identity).
    assert adapter_identity.policy_checkpoint_sha != base_identity.policy_checkpoint_sha
