"""Decode invariant I2: a horizon-limited proof is not a refuted one.

These live outside `test_grammar_fastpath.py` deliberately. That file is a
large legacy suite whose runtime alone exceeds the repository run cap, so
editing it drags every one of its cases into CI's changed-test selection. The
behaviour under test here is `exact_forced_token_id`, not the fast path.
"""

from __future__ import annotations

def test_horizon_limited_domain_falls_through_to_the_dfa_proof() -> None:
    """I2: a budget too short to enumerate a witness is not a contradiction."""
    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        build_completion_forest,
    )
    from slm_training.dsl.grammar.fastpath.engine import engine_for_dsl
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
    from slm_training.models.grammar import GrammarDecodeState, exact_forced_token_id

    tok = DSLNativeTokenizer.build()
    prefix = [tok.bos_id, *tok.encode("root", add_special=False)]
    equals = tok.token_to_id["="]

    # A generous horizon lets the compiler enumerate a witness and prove the
    # singleton directly.
    generous = GrammarDecodeState(engine=engine_for_dsl("openui"))
    generous.remaining_tokens = 32
    assert build_completion_forest(
        tok, prefix, remaining_tokens=32
    ).coverage == "complete"
    assert exact_forced_token_id(tok, prefix, state=generous) == equals

    # A short horizon cannot, but the DFA has already decided. The bypass must
    # agree with the generous case rather than spend a forward.
    tight = GrammarDecodeState(engine=engine_for_dsl("openui"))
    tight.remaining_tokens = 1
    assert build_completion_forest(tok, prefix, remaining_tokens=1).coverage != "complete"
    assert exact_forced_token_id(tok, prefix, state=tight) == equals


def test_complete_domain_with_several_candidates_still_refuses() -> None:
    """A complete domain is authoritative in both directions."""
    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        build_completion_forest,
    )
    from slm_training.dsl.grammar.fastpath.engine import engine_for_dsl
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
    from slm_training.models.grammar import GrammarDecodeState, exact_forced_token_id

    tok = DSLNativeTokenizer.build()
    prefix = tok.encode("root = ", add_special=False)
    state = GrammarDecodeState(engine=engine_for_dsl("openui"))
    state.remaining_tokens = 32
    forest = build_completion_forest(tok, prefix, remaining_tokens=32)
    if forest.coverage == "complete" and len(set(forest.candidate_ids)) > 1:
        assert exact_forced_token_id(tok, prefix, state=state) is None


def test_exact_forced_token_id_charges_compiler_ms_not_unattributed(
    monkeypatch,
) -> None:
    """Decode-cost metering: this grammar-authority call must be attributed.

    ``exact_forced_token_id`` (the legacy/non-compiler-tree LTR decode's I2
    bypass) calls the exact same expensive ``build_completion_forest`` grammar
    authority as the compiler-tree path's timed forest build. Before this
    fix, this call site had zero ``timed_ms`` instrumentation, so a checkpoint
    that decodes through this mechanism (``compiler_decode_mode="off"``, the
    checkpoint's own persisted config) could spend most of its real decode
    wall time here while ``compiler_ms`` stayed at 0.0 and the cost silently
    fell into ``unattributed_ms`` -- making ``compiler_ms_mean`` incomparable
    across checkpoints/cycles that happen to decode via different mechanisms
    (see docs/design/compiler-tree-forced-closure-decode-metering-gap.md).
    """
    import time

    from slm_training.dsl.grammar.fastpath import compiler_draft
    from slm_training.dsl.grammar.fastpath.engine import engine_for_dsl
    from slm_training.models import decode_stats
    from slm_training.models.decode_stats import DecodeStats
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
    from slm_training.models.grammar import GrammarDecodeState, exact_forced_token_id

    tokenizer = DSLNativeTokenizer.build()
    prefix = [tokenizer.bos_id, *tokenizer.encode("root", add_special=False)]
    state = GrammarDecodeState(engine=engine_for_dsl("openui"))
    state.remaining_tokens = 32

    sleep_seconds = 0.05

    # Delay the real grammar authority rather than replacing its result --
    # the test pins that whatever this call actually costs lands in
    # compiler_ms, not a specific return value.
    real_build = compiler_draft.build_completion_forest

    def delayed_build(*args, **kwargs):
        time.sleep(sleep_seconds)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(compiler_draft, "build_completion_forest", delayed_build)

    stats = DecodeStats()
    previous = decode_stats.set_active_stats(stats)
    try:
        exact_forced_token_id(tokenizer, prefix, state=state)
    finally:
        decode_stats.set_active_stats(previous)

    assert stats.compiler_ms >= sleep_seconds * 1000 * 0.9, (
        "exact_forced_token_id's build_completion_forest call must be timed "
        "into compiler_ms, not silently dropped into unattributed_ms"
    )
