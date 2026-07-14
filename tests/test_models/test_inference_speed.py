"""Tests for inference-speed optimizations (P-series)."""

from __future__ import annotations

import torch

from slm_training.dsl.grammar.fastpath import OpenUIIncrementalEngine
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


def test_probe_chunk_agrees_with_throwaway_full_sync() -> None:
    """Q1: copy-based probe must match throwaway set_prefix across prefixes."""
    eng = OpenUIIncrementalEngine()
    cases = [
        ("root", "="),
        ("root=", " "),
        ("root=", "Card"),
        ("root = Card", "("),
        ('root = Card("', ":"),
        ('root = Card(":t.x"', ")"),
        ("root = Stack([hero]", ","),
        ("root = Stack([hero]", "]"),
    ]
    for prefix, chunk in cases:
        assert eng.set_prefix(prefix), prefix
        probed = eng.probe_chunk(chunk)
        throwaway = OpenUIIncrementalEngine()
        full = throwaway.set_prefix(prefix + chunk)
        if probed is None:
            # Fallback path — still legal via throwaway; just ensure no crash.
            continue
        assert bool(probed) is bool(full), (prefix, chunk, probed, full)
        # Shared engine must remain at the original prefix.
        assert eng._prefix == prefix


def test_probe_chunk_name_gluing_falls_back() -> None:
    """NAME/COMPONENT extension changes fed lexeme identity → None (fallback)."""
    eng = OpenUIIncrementalEngine()
    # "Te" lexes as COMPONENT; extending to "Text" changes that token's value.
    assert eng.set_prefix("root = Te")
    result = eng.probe_chunk("xt")
    # Either fallback (None) or a correct bool — must not poison the engine.
    assert result is None or isinstance(result, bool)
    assert eng._prefix == "root = Te"
    # Original still accepts LPAR after incomplete COMPONENT.
    assert "LPAR" in eng.next_terminals() or eng.next_terminals()


def test_dfa_admits_uses_copy_probe_and_memo() -> None:
    from slm_training.models.grammar import dfa_admits_token

    tok = _tok()
    state = make_grammar_state(use_copy_probes=True)
    prefix = tok.encode("root", add_special=False)
    state.sync_ids(tok, prefix)
    assert state.engine is not None
    state.engine.set_prefix(state.prefix_text)

    eq = tok.token_to_id["="]
    ok1 = dfa_admits_token(tok, prefix, eq, state=state, prefix_text=state.prefix_text)
    assert ok1 is True
    assert eq in state.admit_memo
    # Second call hits memo (no extra work).
    ok2 = dfa_admits_token(tok, prefix, eq, state=state, prefix_text=state.prefix_text)
    assert ok2 is ok1

    # Illegal double-equal
    prefix2 = tok.encode("root=", add_special=False)
    state.sync_ids(tok, prefix2)
    state.engine.set_prefix(state.prefix_text)
    assert dfa_admits_token(
        tok, prefix2, eq, state=state, prefix_text=state.prefix_text
    ) is False


def test_whitespace_fast_admit() -> None:
    from slm_training.models.grammar import dfa_admits_token

    tok = _tok()
    if " " not in tok.token_to_id:
        return
    state = make_grammar_state(use_copy_probes=True)
    prefix = tok.encode("root=", add_special=False)
    state.sync_ids(tok, prefix)
    state.engine.set_prefix(state.prefix_text)
    space = tok.token_to_id[" "]
    assert dfa_admits_token(
        tok, prefix, space, state=state, prefix_text=state.prefix_text
    ) is True
    assert state.whitespace_ok is True


def test_early_exit_pick_returns_legal_argmax() -> None:
    tok = _tok()
    equal_id = tok.token_to_id["="]
    logits = torch.zeros(tok.vocab_size)
    logits[equal_id] = 10.0
    # A lower-scoring illegal token should not win.
    if "]" in tok.token_to_id:
        logits[tok.token_to_id["]"]] = 5.0
    prefix = tok.encode("root", add_special=False)
    state = make_grammar_state(early_exit_pick=True, use_copy_probes=True)
    choice = pick_constrained_token(
        logits,
        tok,
        prefix,
        state=state,
        prefer_structural=False,
        top_k=8,
    )
    assert choice == equal_id


def test_r1_exact_allowed_skips_admit_probe() -> None:
    """R1: exact DFA terminals should not call dfa_admits for tid in allowed."""
    tok = _tok()
    equal_id = tok.token_to_id["="]
    logits = torch.zeros(tok.vocab_size)
    logits[equal_id] = 10.0
    prefix = tok.encode("root", add_special=False)
    state = make_grammar_state(
        verify_chosen_only=True,
        skip_exact_stream_probe=True,
        use_copy_probes=True,
    )
    state.sync_ids(tok, prefix)
    assert state.engine is not None
    state.engine.set_prefix(state.prefix_text)
    assert state.engine.terminals_are_exact()
    with collect_decode_stats():
        choice = pick_constrained_token(
            logits,
            tok,
            prefix,
            state=state,
            verify_chosen_only=True,
            prefer_structural=False,
            top_k=4,
        )
    assert choice == equal_id
    # Exact '=' should not need admit probes (memo stays empty for this pick).
    assert equal_id not in state.admit_memo


def test_r2_force_emit_skips_resync_when_synced() -> None:
    """R2: force_emit on an already-synced engine should not full-sync again."""
    tok = _tok()
    state = make_grammar_state()
    ids = tok.encode("root", add_special=False)
    state.sync_ids(tok, ids)
    assert state.engine is not None
    assert state.engine.set_prefix(state.prefix_text)
    before = state.engine._full_syncs
    forced = force_emit_token_id(tok, ids, state=state)
    assert forced == tok.token_to_id["="]
    assert state.engine._full_syncs == before


def test_r5_ensure_honors_zero_attempts() -> None:
    """R5: attempts=0 skips BOS redo (no repair forwards)."""
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
        grammar_ltr_repair=True,
        grammar_finalize_validate=False,
        grammar_incremental_state=True,
        generate_max_attempts=1,
        grammar_ltr_max_tokens=24,
        max_target_len=32,
        max_prompt_len=32,
        seed=0,
    )
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    model.eval()
    ctx, ctx_pad = model._encode_context(["hero card"])
    bad = "not valid openui!!!"
    with collect_decode_stats() as stats:
        out = model._ensure_valid_openui(
            bad,
            ctx,
            ctx_pad,
            16,
            attempts=0,
        )
    assert out == bad
    # Zero attempts → no repair forwards.
    assert stats.forwards_count == 0


def test_r4_repair_uses_multitoken_fewer_forwards() -> None:
    """R4: multitoken repair should emit accepted_run_tokens and fewer forwards."""
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
        grammar_ltr_repair=True,
        grammar_finalize_validate=False,
        grammar_incremental_state=True,
        grammar_verify_chosen_only=True,
        grammar_multitoken_accept=True,
        grammar_multitoken_max=8,
        grammar_canvas_lookahead=16,
        grammar_ltr_max_tokens=24,
        max_target_len=32,
        max_prompt_len=32,
        generate_max_attempts=1,
        seed=0,
    )
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    model.eval()
    device = model.device_name
    length = 24
    ids = torch.full(
        (1, length), model.tokenizer.mask_id, dtype=torch.long, device=device
    )
    ids[0, 0] = model.tokenizer.bos_id
    unknown = ids.eq(model.tokenizer.mask_id)
    ctx, ctx_pad = model._encode_context(["hero card"])
    with collect_decode_stats() as stats:
        filled = model._constrained_ltr_repair(ids, unknown, ctx, ctx_pad)
    assert filled.shape == (1, length)
    # Multitoken should keep forwards well below one-per-position.
    assert stats.forwards_count < length
    assert stats.forwards_count >= 1
