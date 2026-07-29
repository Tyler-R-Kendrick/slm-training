"""Structural reuse invariant: forked-engine forests == fresh-engine forests.

``_openui_completion_domain``'s tail-witness recursion forks the parent
node's ``OpenUIIncrementalEngine`` (``copy()``) and lets the child build take
the incremental sync path instead of re-lexing+re-feeding the whole prefix.
The forest must be bit-identical to a from-scratch build for the same prefix
— reuse is a pure cache layer, never a change to the legal set.
"""

from __future__ import annotations

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    _build_openui_completion_forest,
)
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

_PREFIXES = (
    "root",
    "root =",
    "root = TextContent",
    'root = TextContent(":slot_0")',
    'root = TextContent(":slot_0")\n',
)


def _tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


def test_engine_copy_shares_position_not_parser_state() -> None:
    engine = OpenUIIncrementalEngine()
    assert engine.set_prefix("root =")
    fork = engine.copy()
    # The fork is synced to the same prefix and sees the same accepts…
    parent_accepts = engine.next_terminals()
    assert fork.next_terminals() == parent_accepts
    # …but advances independently of the original.
    assert fork.advance("TextContent")
    assert engine.next_terminals() == parent_accepts


def test_forked_engine_forest_equals_fresh_forest() -> None:
    tok = _tok()
    engine = OpenUIIncrementalEngine()
    for text in _PREFIXES:
        ids = list(tok.encode(text, add_special=False))
        fresh = _build_openui_completion_forest(
            tok, ids, slot_contract=["slot_0"]
        )
        reused = _build_openui_completion_forest(
            tok, ids, engine=engine, slot_contract=["slot_0"]
        )
        assert reused == fresh, f"forked-engine forest diverged at {text!r}"
        # Fork the just-synced engine so the next (extended) prefix takes the
        # incremental path, mirroring the tail-witness recursion in pack.py.
        engine = engine.copy()


def test_advance_checked_matches_probe_then_advance() -> None:
    # For a chain of pieces (including gluing + whitespace), advance_checked
    # must land in the same accepts state as the probe_chunk + advance pair.
    pieces = ["root", " ", "=", " Text", "Content", "(", '"', ":slot_0", '"', ")", "\n"]
    reference = OpenUIIncrementalEngine()
    checked = OpenUIIncrementalEngine()
    for piece in pieces:
        probe = reference.probe_chunk(piece)
        if probe is None:
            expected = reference.set_prefix(reference._prefix + piece)
        elif probe:
            expected = reference.advance(piece)
        else:
            expected = False
        actual = checked.advance_checked(piece)
        assert actual == expected, f"outcome diverged at piece {piece!r}"
        assert checked.next_terminals() == reference.next_terminals(), (
            f"accepts diverged at piece {piece!r}"
        )


def test_incremental_sync_counters_show_reuse() -> None:
    tok = _tok()
    engine = OpenUIIncrementalEngine()
    engines = [engine]
    for text in _PREFIXES:
        ids = list(tok.encode(text, add_special=False))
        _build_openui_completion_forest(
            tok, ids, engine=engine, slot_contract=["slot_0"]
        )
        engine = engine.copy()
        engines.append(engine)
    # First build is a full sync; every later build on the extended prefix
    # must reuse the forked parser state instead of re-feeding from scratch.
    # (Counters live per instance, so sum across the fork chain.)
    assert sum(eng._full_syncs for eng in engines) <= 1
    # At least one extension step may collapse to a same-text no-op (a trailing
    # NL adds no fed lexeme), so require reuse on the majority of steps.
    assert sum(eng._incremental_advances for eng in engines) >= len(_PREFIXES) - 2
