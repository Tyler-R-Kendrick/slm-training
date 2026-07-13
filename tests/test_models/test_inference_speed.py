"""Tests for inference-speed optimizations (P-series)."""

from __future__ import annotations

import torch

from slm_training.grammar_fastpath import OpenUIIncrementalEngine
from slm_training.models.decode_stats import DecodeStats, collect_decode_stats
from slm_training.models.grammar import (
    force_emit_token_id,
    make_grammar_state,
    pick_constrained_token,
)
from slm_training.models.tokenizer import OpenUITokenizer
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.dsl.schema import ExampleRecord

SAMPLE = 'root = Card(":t.x")\n'


def _tok() -> OpenUITokenizer:
    return OpenUITokenizer.build(
        [SAMPLE, 'root = Row(":j")\n', "root = Stack([hero])\n"]
    )


def test_engine_incremental_advance_matches_full_sync() -> None:
    eng = OpenUIIncrementalEngine()
    assert eng.set_prefix("root")
    assert eng.is_deterministic_next() == "="
    assert eng.advance("=")
    assert eng._prefix == "root="
    # Fresh engine with full sync should agree on accepts.
    eng2 = OpenUIIncrementalEngine()
    assert eng2.set_prefix("root=")
    assert eng.next_terminals() == eng2.next_terminals()
    assert eng._incremental_advances >= 1
    assert eng._full_syncs >= 1


def test_engine_terminals_are_exact_for_structural() -> None:
    eng = OpenUIIncrementalEngine()
    assert eng.set_prefix("root")
    assert eng.terminals_are_exact() is True
    assert eng.set_prefix("root=")
    # After '=' we expect COMPONENT (broad) among accepts.
    assert eng.terminals_are_exact() is False


def test_grammar_state_reuses_prefix_text() -> None:
    tok = _tok()
    state = make_grammar_state()
    ids = tok.encode("root", add_special=False)
    text = state.sync_ids(tok, ids)
    assert "root" in text
    # Append '=' via advance_token
    eq = tok.token_to_id["="]
    state.advance_token(tok, eq)
    assert state.prefix_text.endswith("=")
    assert len(state.prefix_ids) == len(ids) + 1


def test_force_emit_with_state() -> None:
    tok = _tok()
    state = make_grammar_state()
    ids = tok.encode("root", add_special=False)
    forced = force_emit_token_id(tok, ids, state=state)
    assert forced is not None
    assert tok.id_to_token[forced] == "="


def test_verify_chosen_only_accepts_legal_argmax() -> None:
    tok = _tok()
    equal_id = tok.token_to_id["="]
    logits = torch.zeros(tok.vocab_size)
    logits[equal_id] = 10.0
    prefix = tok.encode("root", add_special=False)
    state = make_grammar_state(verify_chosen_only=True)
    choice = pick_constrained_token(
        logits,
        tok,
        prefix,
        state=state,
        verify_chosen_only=True,
        prefer_structural=False,
    )
    assert choice == equal_id


def test_decode_stats_collect_during_pick() -> None:
    tok = _tok()
    logits = torch.zeros(tok.vocab_size)
    logits[tok.token_to_id["="]] = 5.0
    prefix = tok.encode("root", add_special=False)
    with collect_decode_stats() as stats:
        pick_constrained_token(logits, tok, prefix, top_k=4)
    assert isinstance(stats, DecodeStats)
    assert stats.pick_ms >= 0.0
    assert stats.dfa_sync_count >= 1


def test_ltr_incremental_generates() -> None:
    records = [
        ExampleRecord(
            id="t1",
            prompt="hero card",
            openui=SAMPLE,
            design_md="# Design\n",
            split="train",
            source="fixture",
        )
    ]
    cfg = TwoTowerConfig(
        context_backend="scratch",
        d_model=64,
        n_heads=2,
        context_layers=1,
        denoiser_layers=2,
        grammar_constrained=True,
        grammar_ltr_primary=True,
        grammar_ltr_repair=False,
        grammar_finalize_validate=False,
        grammar_incremental_state=True,
        grammar_ltr_max_tokens=24,
        max_target_len=32,
        max_prompt_len=32,
        seed=0,
    )
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    model.eval()
    text, stats = model.generate_with_stats("hero card")
    assert isinstance(text, str)
    assert stats.forwards_count >= 1
    assert stats.total_ms > 0


def test_multitoken_and_lookahead_flags_smoke() -> None:
    records = [
        ExampleRecord(
            id="t1",
            prompt="hero card",
            openui=SAMPLE,
            design_md="# Design\n",
            split="train",
            source="fixture",
            placeholders=[":t.x"],
        )
    ]
    cfg = TwoTowerConfig(
        context_backend="scratch",
        d_model=64,
        n_heads=2,
        context_layers=1,
        denoiser_layers=2,
        grammar_constrained=True,
        grammar_ltr_primary=True,
        grammar_ltr_repair=False,
        grammar_finalize_validate=False,
        grammar_incremental_state=True,
        grammar_multitoken_accept=True,
        grammar_multitoken_max=4,
        grammar_canvas_lookahead=16,
        grammar_ltr_max_tokens=24,
        max_target_len=32,
        max_prompt_len=32,
        seed=0,
    )
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    model.eval()
    text = model.generate("hero card")
    assert isinstance(text, str)


def test_p1_bitexact_vs_legacy_on_force_path() -> None:
    """Force-emit path with/without state should agree on the forced '='."""
    tok = _tok()
    ids = tok.encode("root", add_special=False)
    a = force_emit_token_id(tok, ids, state=None)
    state = make_grammar_state()
    b = force_emit_token_id(tok, ids, state=state)
    assert a == b == tok.token_to_id["="]
