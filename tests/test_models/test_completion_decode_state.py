from __future__ import annotations

from types import SimpleNamespace

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)
from slm_training.dsl.pack import _openui_completion_domain
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.grammar import CompletionBatchCache, GrammarDecodeState


class _Tokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 9
    mask_id = 8
    id_to_token = {1: "", 2: "=", 3: "root"}
    token_to_id = {"=": 2, "root": 3}

    def decode(self, ids, **_kwargs):
        return "".join(self.id_to_token.get(int(token_id), "") for token_id in ids)


class _Session:
    def __init__(self) -> None:
        self.counters = {"session_starts": 1, "unique_states": 1}
        self.restored: list[int] = []

    def stats(self):
        return dict(self.counters)

    def advance(self, state_id: int, token_id: int):
        self.counters["transition_cache_misses"] = (
            self.counters.get("transition_cache_misses", 0) + 1
        )
        return state_id + token_id

    def restore(self, state_id: int):
        self.restored.append(state_id)
        return state_id

    def forced_closure(self, state_id: int, room: int):
        return ((3,) if room else (), state_id + 3, "complete")


def test_row_completion_state_advances_and_rollback_restores_interned_id() -> None:
    tok = _Tokenizer()
    session = _Session()
    state = GrammarDecodeState(prefix_ids=[1], prefix_text="")
    with collect_decode_stats() as stats:
        state.bind_completion_session(session, ("authority",), 10, (1,))
        state.advance_token(tok, 2)
        assert state.completion_state_id == 12
        assert state.completion_prefix_states[(1, 2)] == 12
        state.sync_ids(tok, [1])

    assert state.completion_state_id == 10
    assert session.restored == [10]
    assert stats.completion_session_starts == 1
    assert stats.completion_transition_cache_misses == 1


def test_forced_closure_uses_current_completion_state() -> None:
    session = _Session()
    state = GrammarDecodeState(prefix_ids=[1])
    state.bind_completion_session(session, ("authority",), 10, (1,))
    assert state.completion_forced_closure(4) == ((3,), 13, "complete")


def test_equivalent_batch_rows_share_domain_but_own_sessions(monkeypatch) -> None:
    class Session:
        def __init__(self, tokenizer, **_kwargs):
            self.tokenizer = tokenizer
            self.prefix = ()

        def stats(self):
            return {"session_starts": 1, "unique_states": 1}

        def seed(self, prefix, **_kwargs):
            self.prefix = tuple(prefix)
            return 0

        def prefix_ids_of(self, _state_id):
            return self.prefix

        def outgoing(self, _state_id):
            return CompletionForest(
                (CompletionPath((self.tokenizer.eos_id,), "eos"),), "complete"
            )

    import slm_training.dsl.grammar.fastpath.completion_kernel as kernel

    monkeypatch.setattr(kernel, "CompletionSession", Session)
    tok = _Tokenizer()
    tok.kind_ids = lambda *_args: set()
    shared = CompletionBatchCache()
    states = [GrammarDecodeState(completion_batch_cache=shared) for _ in range(2)]

    def request(state):
        return SimpleNamespace(
            tokenizer=tok,
            prefix_ids=(tok.bos_id,),
            runtime_symbols=(),
            remaining_tokens=4,
            state=state,
            slot_contract=(),
            max_path_tokens=8,
            min_content=0,
            explain=False,
        )

    with collect_decode_stats() as stats:
        first = _openui_completion_domain(request(states[0]))
        second = _openui_completion_domain(request(states[1]))

    assert first is second
    assert states[0].completion_session is not states[1].completion_session
    assert stats.completion_shared_domain_misses == 1
    assert stats.completion_shared_domain_hits == 1

    import slm_training.models.decode_stats as decode_stats

    def _expired() -> None:
        raise TimeoutError("decode deadline")

    monkeypatch.setattr(decode_stats, "check_decode_deadline", _expired)
    with pytest.raises(TimeoutError):
        _openui_completion_domain(request(states[1]))


def test_mutated_macro_authority_replaces_row_session(monkeypatch) -> None:
    class Session:
        def __init__(self, tokenizer, **_kwargs):
            self.tokenizer = tokenizer
            self.prefix = ()

        def stats(self):
            return {"session_starts": 1, "unique_states": 1}

        def seed(self, prefix, **_kwargs):
            self.prefix = tuple(prefix)
            return 0

        def prefix_ids_of(self, _state_id):
            return self.prefix

        def outgoing(self, _state_id):
            return CompletionForest(
                (CompletionPath((self.tokenizer.eos_id,), "eos"),), "complete"
            )

    import slm_training.dsl.grammar.fastpath.completion_kernel as kernel

    monkeypatch.setattr(kernel, "CompletionSession", Session)
    tok = _Tokenizer()
    tok.kind_ids = lambda *_args: set()
    tok.macro_expansions = ()
    state = GrammarDecodeState()
    request = SimpleNamespace(
        tokenizer=tok,
        prefix_ids=(tok.bos_id,),
        runtime_symbols=(),
        remaining_tokens=4,
        state=state,
        slot_contract=(),
        max_path_tokens=8,
        min_content=0,
        explain=False,
    )

    first = _openui_completion_domain(request)
    first_session = state.completion_session
    tok.macro_expansions = (("Card", "(", ")"),)
    second = _openui_completion_domain(request)

    assert first is not second
    assert state.completion_session is not first_session
