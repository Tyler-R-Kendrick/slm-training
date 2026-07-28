"""Regression tests for the tail-only re-lex in
``OpenUIIncrementalEngine._incremental_sync``
(docs/design/decode-compiler-tree-incremental-sync-relex-finding.md).

That finding showed ``_incremental_sync`` called ``_lex_tokens(prefix)`` with
the *entire current prefix* on every call, even though it already tracks
``_fed_token_count`` and only feeds the newly completed tokens to the
``InteractiveParser``. Re-lexing the whole prefix on every incremental step
makes a growing decode session cost O(n^2) in the prefix length instead of
O(n): measured 81.2x more characters re-lexed than the theoretical O(n)
minimum on a realistic 326-char full OpenUI record, and 2027us/char^2 which
stayed flat with length (quadratic evidence).

The fix tracks ``_settle_pos`` -- the start offset of the last fed token --
and re-lexes only ``prefix[_settle_pos:]`` (the "unsettled tail": the
possibly-still-growing last token plus newly appended text). Every token
before that offset is provably final because ``openui.lark`` has no
lookbehind (``\\b``/``(?<=``/``(?<!``) terminals: a regex-based lexer decides
where a token ends using only the characters up to that boundary, so
appending more text after it can never retroactively change an earlier
token's identity.

These tests prove: (a) the incremental (tail-relex) path produces byte
identical ``accepts()``/token-stream/parser state to a shadow engine that
always does a full resync, across growing sessions including NAME/COMPONENT
gluing and malformed-fragment recovery edge cases (I6 proof); (b) the total
characters re-lexed across a growing session stays near-linear, not
quadratic (regression guard for the finding's bug); (c) the fix is
measurably faster on a realistic full-record decode session; and (d)
``build_completion_forest``'s ``coverage``/``n_paths`` are unchanged on the
established finding prefixes (I3/I6: a caching/perf change must never become
a legality change).
"""

from __future__ import annotations

import re
import time

import pytest

from slm_training.dsl.grammar.fastpath import engine as engine_mod
from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine

# Exact prefixes from PR #1171's dead_end_trace / the lexer-rebuild finding.
PREFIX8 = "root = Card([b1"
PREFIX9 = "root = Card([b1,"

_CHUNK_RE = re.compile(r"\w+|\s+|.")

FULL_RECORD = (
    "root = Stack([layer8], 'row')\n"
    "buttons1 = Buttons([])\n"
    "form2 = Form('$0', buttons1, [])\n"
    "textcontent3 = TextContent(':gen1_textcontent_plain.text')\n"
    "input4 = Input('$1')\n"
    "layer5 = Stack([form2, textcontent3, input4], 'column')\n"
    "layer6 = Stack([layer5], 'column')\n"
    "layer7 = Stack([layer6], 'column')\n"
    "layer8 = Stack([layer7], 'column')"
)

SESSIONS = {
    "prefix8_char_by_char": list(PREFIX8),
    "prefix9_char_by_char": list(PREFIX9),
    "full_record_word_chunked": _CHUNK_RE.findall(FULL_RECORD),
    "gluing_component_name": list("root = Te") + list("xtContent("),
    "gluing_number_after_comment": list("root = Card([b1]) # ") + list("123"),
    "trailing_whitespace_runs": ["root", " ", " ", "=", " ", "Card", "(", "["],
    "malformed_then_recover": list("root = C@@@") + list(" ard(["),
}


@pytest.mark.parametrize("chunks", SESSIONS.values(), ids=SESSIONS.keys())
def test_incremental_sync_matches_full_resync_across_sessions(chunks: list[str]) -> None:
    """I6 proof: the tail-relex incremental path must reach the exact same
    parser state (accepts(), fed tokens, prefix) as always doing a full
    resync, at every step of a growing session."""
    real = OpenUIIncrementalEngine()
    real.reset()
    shadow = OpenUIIncrementalEngine()
    shadow.reset()
    acc = ""
    for chunk in chunks:
        acc += chunk
        ok_real = real.advance(chunk)
        ok_shadow = shadow._full_sync(acc)
        assert ok_real == ok_shadow
        assert real.next_terminals() == shadow.next_terminals()
        assert real._fed_tokens == shadow._fed_tokens
        assert real._prefix == shadow._prefix == acc


def test_incremental_sync_relexes_near_linear_not_quadratic() -> None:
    """Regression guard for the finding's bug: total characters re-lexed
    across a growing session must stay close to the O(n) minimum (sum of
    chunk lengths), not blow up to O(n^2) (the pre-fix behavior measured an
    81.2x amplification over O(n) on this exact record)."""
    chunks = _CHUNK_RE.findall(FULL_RECORD)
    lexed_lengths: list[int] = []
    orig_lex = OpenUIIncrementalEngine._lex

    def counting_lex(self, text):
        lexed_lengths.append(len(text))
        return orig_lex(self, text)

    OpenUIIncrementalEngine._lex = counting_lex
    try:
        eng = OpenUIIncrementalEngine()
        eng.reset()
        for chunk in chunks:
            eng.advance(chunk)
    finally:
        OpenUIIncrementalEngine._lex = orig_lex

    total_relexed = sum(lexed_lengths)
    ideal = sum(len(c) for c in chunks)
    amplification = total_relexed / ideal
    # Pre-fix amplification measured 81.2x for this record; post-fix
    # measured 2.8x (each still-growing token's tail gets re-lexed a few
    # times before it settles). 8x is a generous ceiling, robust to grammar
    # tweaks, that still clearly rules out the O(n^2) regression.
    assert amplification < 8.0, (
        f"incremental sync re-lexed {total_relexed} chars total for a "
        f"{ideal}-char ideal (O(n)) session -- {amplification:.1f}x "
        "amplification, consistent with a return to whole-prefix re-lex"
    )


def test_incremental_sync_faster_than_always_full_resync_on_long_session() -> None:
    """Wall-time benchmark: a long growing session (the 326-char full record,
    141 growing prefixes) must be measurably faster through the tail-relex
    incremental path than an engine forced to full-resync at every step.
    Threshold (1.3x) is well under the ~2.2x measured locally, to stay
    robust on a slower/loaded CI host."""
    chunks = _CHUNK_RE.findall(FULL_RECORD)

    real = OpenUIIncrementalEngine()
    real.reset()
    t0 = time.perf_counter()
    for chunk in chunks:
        real.advance(chunk)
    incremental_s = time.perf_counter() - t0

    shadow = OpenUIIncrementalEngine()
    shadow.reset()
    acc = ""
    t0 = time.perf_counter()
    for chunk in chunks:
        acc += chunk
        shadow._full_sync(acc)
    full_resync_s = time.perf_counter() - t0

    assert incremental_s < full_resync_s / 1.3, (
        f"incremental path ({incremental_s:.4f}s) did not clear the 1.3x "
        f"speedup floor over always-full-resync ({full_resync_s:.4f}s)"
    )


def test_settle_pos_resets_on_reset_and_new_engine() -> None:
    """``_settle_pos`` must start at 0 for a fresh engine/session -- the
    first sync after ``reset()`` always has nothing settled yet."""
    eng = OpenUIIncrementalEngine()
    assert eng._settle_pos == 0
    eng.reset()
    assert eng._settle_pos == 0
    eng.advance("root = Card([b1")
    assert eng._settle_pos > 0
    eng.reset()
    assert eng._settle_pos == 0


@pytest.mark.parametrize(
    ("prefix", "expected_n_paths"),
    [
        (PREFIX8, 2),
        (PREFIX9, 12),
    ],
)
def test_build_completion_forest_unchanged_on_finding_prefixes(
    prefix: str, expected_n_paths: int
) -> None:
    """I3/I6 decode-invariant proof: on the exact prefixes measured in the
    lexer-rebuild finding, ``build_completion_forest`` (which drives
    ``_incremental_sync`` internally via the recursive witness search) still
    returns the same ``coverage``/``n_paths`` after the tail-relex fix. A
    caching/perf change must never become a legality change."""
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    prefix_ids = [tok.bos_id, *tok.encode(prefix, add_special=False)]
    forest = build_completion_forest(
        tok, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
    )
    assert forest.coverage == "complete"
    assert len(forest.paths) == expected_n_paths


def test_load_lexer_still_cached_per_grammar_path() -> None:
    """The incremental-sync fix reuses ``_lex``/``_load_lexer`` unchanged --
    guard against accidentally reintroducing a per-call lexer rebuild while
    reworking ``_incremental_sync``."""
    path = str((engine_mod.GRAMMARS_DIR / "openui.lark").resolve())
    assert engine_mod._load_lexer(path) is engine_mod._load_lexer(path)
