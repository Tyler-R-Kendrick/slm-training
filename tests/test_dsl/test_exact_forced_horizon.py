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
