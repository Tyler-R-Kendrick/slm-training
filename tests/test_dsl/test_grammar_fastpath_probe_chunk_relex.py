"""Regression tests for the tail-only re-lex in
``OpenUIIncrementalEngine.probe_chunk``
(docs/design/decode-compiler-tree-probe-chunk-relex-finding.md).

`_incremental_sync`'s fix
(docs/design/decode-compiler-tree-incremental-sync-relex-finding.md) left
`probe_chunk` (the Q1 copy-probe path) explicitly as a "next step": it has
the exact same shape as pre-fix `_incremental_sync` -- `_lex_tokens(self._prefix
+ chunk)` re-lexes the *entire* current prefix plus the probed chunk on
every call, even though the probe never mutates engine state and could reuse
the same `_settle_pos` boundary `_incremental_sync` already tracks.

The fix mirrors `_incremental_sync`'s: re-lex only
``new_prefix[self._settle_pos:]`` instead of the whole prefix, splice the
tail tokens onto the already-known ``self._fed_tokens[:prev_settled_count]``,
and leave ``self._settle_pos``/``self._fed_token_count`` untouched (the probe
must stay read-only).

These tests prove: (a) the tail-relex probe returns byte-identical
``bool | None`` results to a shadow implementation that always fully relexes
``new_prefix`` (I6 proof), across growing sessions and representative probe
chunks including gluing edge cases; (b) the total characters re-lexed stays
near-linear, not quadratic (regression guard); (c) the fix is measurably
faster; and (d) ``build_completion_forest`` (which calls ``probe_chunk``
internally via the I3 witness search in `compiler_draft.py`) still returns
unchanged ``coverage``/``n_paths`` on the established finding prefixes.
"""

from __future__ import annotations

import re
import time

import pytest
from lark.exceptions import UnexpectedEOF, UnexpectedToken

from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine

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

PROBES = ["x", "Text", "(", ")", ",", " ", "1", "abc)", "Content("]

SESSIONS = {
    "prefix8_char_by_char": list(PREFIX8),
    "prefix9_char_by_char": list(PREFIX9),
    "full_record_word_chunked": _CHUNK_RE.findall(FULL_RECORD),
    "gluing_component_name": list("root = Te") + list("xtContent("),
}


def _shadow_probe_chunk(eng: OpenUIIncrementalEngine, chunk: str) -> bool | None:
    """Pre-fix ``probe_chunk`` semantics: full relex of ``new_prefix`` on
    every call, kept here only as an independent oracle for the fix."""
    if eng._ip is None:
        return None
    if not chunk:
        return True
    new_prefix = eng._prefix + chunk
    tokens = eng._lex_tokens(new_prefix)
    if tokens is None:
        return False
    keys = eng._token_keys(tokens)
    if len(tokens) < eng._fed_token_count or keys[: eng._fed_token_count] != eng._fed_tokens:
        return None
    delta = tokens[eng._fed_token_count :]
    if not delta:
        return True
    try:
        snap = eng._ip.copy()
    except Exception:
        return None
    try:
        for tok in delta:
            snap.feed_token(tok)
    except UnexpectedToken:
        return False
    except UnexpectedEOF:
        pass
    return True


@pytest.mark.parametrize("chunks", SESSIONS.values(), ids=SESSIONS.keys())
def test_probe_chunk_matches_full_resync_shadow_across_sessions(chunks: list[str]) -> None:
    """I6 proof: at every step of a growing session, and for a battery of
    representative probe chunks, the tail-relex ``probe_chunk`` must return
    the exact same ``bool | None`` as a shadow that always fully relexes
    ``new_prefix`` -- and must never mutate engine state (read-only probe)."""
    eng = OpenUIIncrementalEngine()
    eng.reset()
    for chunk in chunks:
        prefix_before = eng._prefix
        fed_before = list(eng._fed_tokens)
        settle_before = eng._settle_pos
        for probe in PROBES:
            real = eng.probe_chunk(probe)
            shadow = _shadow_probe_chunk(eng, probe)
            assert real == shadow, (
                f"probe_chunk diverged: real={real} shadow={shadow} "
                f"prefix={eng._prefix!r} probe={probe!r}"
            )
            # Read-only: state must be untouched by any probe.
            assert eng._prefix == prefix_before
            assert eng._fed_tokens == fed_before
            assert eng._settle_pos == settle_before
        eng.advance(chunk)


def test_probe_chunk_relexes_near_linear_not_quadratic() -> None:
    """Regression guard: total characters re-lexed across a session that
    interleaves ``probe_chunk`` calls with ``advance`` must stay close to the
    O(n) minimum, not blow up quadratically (36.8x amplification measured
    pre-fix on this exact record)."""
    chunks = _CHUNK_RE.findall(FULL_RECORD)
    probe = "abc"
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
            eng.probe_chunk(probe)
            eng.advance(chunk)
    finally:
        OpenUIIncrementalEngine._lex = orig_lex

    total_relexed = sum(lexed_lengths)
    ideal = sum(len(c) for c in chunks) + len(chunks) * len(probe)
    amplification = total_relexed / ideal
    # Pre-fix amplification measured 36.8x for this record+probe session;
    # post-fix measured 2.6x. 8x is a generous ceiling that still clearly
    # rules out the O(n^2) regression.
    assert amplification < 8.0, (
        f"probe_chunk + advance re-lexed {total_relexed} chars total for a "
        f"{ideal}-char ideal (O(n)) session -- {amplification:.1f}x "
        "amplification, consistent with a return to whole-prefix re-lex"
    )


def test_probe_chunk_faster_than_always_full_resync_on_long_session() -> None:
    """Wall-time benchmark: a session that probes then advances through the
    326-char full record must be measurably faster than always fully
    relexing ``new_prefix`` on every probe. Threshold (1.2x) is well under
    the ~1.65x measured locally, to stay robust on a slower/loaded CI host."""
    chunks = _CHUNK_RE.findall(FULL_RECORD)
    probe = "abc"

    real = OpenUIIncrementalEngine()
    real.reset()
    t0 = time.perf_counter()
    for chunk in chunks:
        real.probe_chunk(probe)
        real.advance(chunk)
    real_s = time.perf_counter() - t0

    shadow = OpenUIIncrementalEngine()
    shadow.reset()
    t0 = time.perf_counter()
    for chunk in chunks:
        _shadow_probe_chunk(shadow, probe)
        shadow.advance(chunk)
    shadow_s = time.perf_counter() - t0

    assert real_s < shadow_s / 1.2, (
        f"tail-relex probe_chunk session ({real_s:.4f}s) did not clear the "
        f"1.2x speedup floor over the full-resync shadow ({shadow_s:.4f}s)"
    )


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
    """I3/I6 decode-invariant proof: ``build_completion_forest`` (which calls
    ``probe_chunk`` internally via ``compiler_draft.py``'s witness search)
    still returns the same ``coverage``/``n_paths`` after the tail-relex fix
    on the established finding prefixes. A caching/perf change must never
    become a legality change."""
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    prefix_ids = [tok.bos_id, *tok.encode(prefix, add_special=False)]
    forest = build_completion_forest(
        tok, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
    )
    assert forest.coverage == "complete"
    assert len(forest.paths) == expected_n_paths
