"""Realistic-chunk-size re-measurement of the `probe_chunk` suffix-anchor fix
(docs/design/decode-compiler-tree-probe-chunk-realistic-chunk-size-finding.md).

PR #1176's `probe_chunk` suffix-anchor fix (docs/design/
decode-compiler-tree-probe-chunk-suffix-lex-fix.md) measured its win on a
synthetic character-by-character drive and flagged, as an explicit open
"next step", that real call sites (`compiler_draft.py`'s
`build_completion_forest` witness search) probe with *tokenizer piece*
chunks via `token_surface_piece`, not single characters -- typically several
characters at once -- which "may shift the anchor-vs-`InteractiveParser.copy()`
-overhead balance measured [t]here".

This module closes that gap: it drives `probe_chunk` with the exact
per-candidate loop shape `build_completion_forest` uses (`compiler_draft.py`,
`_evaluate_candidates`-style branch loop) -- skip `LIT_STR`/`LIT_NUM`/
`LIT_END` marker tokens (the real call site never probes literal-frame
content directly), probe each remaining token's surface piece, `advance()`
on a truthy probe or `set_prefix()` on `None` -- against real
`DSLNativeTokenizer`-encoded token ids, not single characters.
"""

from __future__ import annotations

import time

from slm_training.dsl.grammar.fastpath import engine as engine_mod
from slm_training.dsl.grammar.fastpath.token_map import token_surface_piece
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

# Same 10-line, multi-statement valid OpenUI DSL program as PR #1174/#1176's
# LONG_PROGRAM, with one change: the Slider's first argument is a placeholder
# (":slot_level") instead of a free-form literal ("$level") -- DSLNativeTokenizer
# rejects non-placeholder, non-registered-fixed-string literal bodies
# (`_encode_string_literal` raises `ValueError: free-form output string is
# forbidden`), and this module needs a real encode() to get realistic pieces.
LONG_PROGRAM = (
    'root = Stack([hero, b1, b2, b3, cardRow, footer])\n'
    'hero = Card([TextContent(":slot_hero_title"), TextContent(":slot_hero_sub")])\n'
    'b1 = Button(":action_one")\n'
    'b2 = Button(":action_two")\n'
    'b3 = Button(":action_three")\n'
    'cardRow = Row([cardA, cardB, cardC])\n'
    'cardA = Card([TextContent(":slot_a_title"), Buttons([b1, b2])])\n'
    'cardB = Card([TextContent(":slot_b_title"), Input(":second")])\n'
    'cardC = Card([TextContent(":slot_c_title"), Slider(":slot_level", "continuous", -1, 1)])\n'
    'footer = Row([TextContent(":slot_footer_left"), TextContent(":slot_footer_right")])\n'
)

# LIT_STR opens a literal frame, LIT_END closes it, LIT_NUM opens a numeric
# literal frame -- compiler_draft.py's per-candidate loop (build_completion_forest)
# treats all three as frame markers and never calls probe_chunk/advance on
# them directly (see compiler_draft.py's `if raw in {"LIT_STR", "LIT_NUM"}:
# ... continue` / `if raw == "LIT_END": ... continue`). Skipping them here
# reproduces that real call-site shape exactly, rather than a shape this
# module invents.
_LITERAL_FRAME_MARKERS = frozenset({"LIT_STR", "LIT_NUM", "LIT_END"})


def _realistic_chunks() -> list[str]:
    """Tokenizer-piece chunks in the exact shape/order `build_completion_forest`
    feeds to `probe_chunk`/`advance` while walking a single (here: the only)
    admitted candidate sequence for `LONG_PROGRAM`."""
    tokenizer = DSLNativeTokenizer.build()
    token_ids = tokenizer.encode(LONG_PROGRAM, add_special=False)
    chunks = []
    for token_id in token_ids:
        raw = str(tokenizer.id_to_token.get(int(token_id), ""))
        if raw in _LITERAL_FRAME_MARKERS:
            continue
        piece = token_surface_piece(tokenizer, int(token_id))
        if piece:
            chunks.append(piece)
    return chunks


class _CharCountingEngine(engine_mod.OpenUIIncrementalEngine):
    """Counts characters passed into ``_lex_tokens`` -- a direct proxy for
    lex work performed, independent of wall-time noise. Duplicated from
    `test_grammar_fastpath_probe_chunk_suffix_lex.py` rather than imported,
    matching that file's own convention of a small self-contained helper per
    fix/finding module."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.chars_lexed = 0

    def _lex_tokens(self, prefix):
        self.chars_lexed += len(prefix)
        return super()._lex_tokens(prefix)


def _disabled_anchor(self, chunk: str) -> None:
    """Self-contained "before" oracle: forces every probe through the
    pre-existing full-prefix lex path, identical technique PR #1176 used."""
    return None


def _drive_realistic_chunks(eng: engine_mod.OpenUIIncrementalEngine, chunks: list[str]) -> list:
    """Exact probe/commit shape of `compiler_draft.py`'s per-candidate branch
    loop (`build_completion_forest`): probe each chunk against the
    currently-synced prefix without committing, then commit via `advance()`
    on a truthy probe or `set_prefix()` on a `None` (fallback) probe."""
    eng.reset()
    branch_text = ""
    results = []
    for chunk in chunks:
        probe = eng.probe_chunk(chunk)
        results.append(probe)
        if probe is None:
            admitted = eng.set_prefix(branch_text + chunk)
        elif probe:
            admitted = eng.advance(chunk)
        else:
            admitted = False
        assert admitted, f"chunk {chunk!r} rejected mid-drive (probe={probe!r})"
        branch_text += chunk
    return results


def test_realistic_chunks_are_multi_character_not_single_character() -> None:
    """Sanity check on the premise this module tests: `build_completion_forest`'s
    real chunks (tokenizer surface pieces) average several characters each,
    not the single-character chunks PR #1176's synthetic drive used --
    otherwise there would be nothing new to measure here."""
    chunks = _realistic_chunks()
    assert len(chunks) > 50, "expected LONG_PROGRAM to yield a substantial chunk stream"
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    assert avg_len > 1.5, (
        f"expected realistic tokenizer-piece chunks to average >1.5 chars, got {avg_len:.2f} "
        "-- if this regresses to ~1.0, the premise (real chunks are multi-char) no longer holds "
        "and this module's comparison to PR #1176's char-by-char drive is no longer meaningful"
    )


def test_realistic_chunks_relex_fewer_characters_than_full_relex_baseline() -> None:
    """Regression guard: probing LONG_PROGRAM with realistic tokenizer-piece
    chunks must re-lex strictly fewer total characters than the pre-existing
    (forced) full-prefix-relex baseline -- proof the suffix-anchor path is
    actually taken at this chunk granularity, not just at single-char
    granularity."""
    chunks = _realistic_chunks()

    new_eng = _CharCountingEngine()
    _drive_realistic_chunks(new_eng, chunks)

    old_eng = _CharCountingEngine()
    orig = engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor
    engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = _disabled_anchor
    try:
        _drive_realistic_chunks(old_eng, chunks)
    finally:
        engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = orig

    assert new_eng.chars_lexed < old_eng.chars_lexed, (
        f"suffix-anchor path did not reduce chars_lexed at realistic chunk size: "
        f"new={new_eng.chars_lexed} old={old_eng.chars_lexed}"
    )
    # Conservative floor, kept well under the ~7.3x measured locally (see the
    # finding doc) -- realistic chunks reduce lex work by *more*, not less,
    # than PR #1176's 1.98x char-by-char figure, because fewer, larger probe
    # calls mean a smaller fraction of the growing prefix is "still live"
    # relative to what's already permanently fed at any given step.
    assert new_eng.chars_lexed * 4.0 < old_eng.chars_lexed, (
        f"chars_lexed reduction below the 4x floor at realistic chunk size: "
        f"new={new_eng.chars_lexed} old={old_eng.chars_lexed}"
    )


def test_realistic_chunk_probe_matches_forced_full_relex_at_every_step() -> None:
    """I3/I6: `probe_chunk`'s return value must be byte-identical, at every
    step of a realistic-chunk-size probe-then-commit drive (not just the
    final one), between the suffix-anchor path and the pre-existing
    full-prefix-relex baseline."""
    chunks = _realistic_chunks()

    new_eng = engine_mod.OpenUIIncrementalEngine()
    new_trace = _drive_realistic_chunks(new_eng, chunks)

    old_eng = engine_mod.OpenUIIncrementalEngine()
    orig = engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor
    engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = _disabled_anchor
    try:
        old_trace = _drive_realistic_chunks(old_eng, chunks)
    finally:
        engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = orig

    assert len(new_trace) == len(old_trace) == len(chunks)
    mismatches = [
        i for i, (a, b) in enumerate(zip(new_trace, old_trace, strict=True)) if a != b
    ]
    assert not mismatches, (
        f"probe_chunk diverged at realistic-chunk steps {mismatches[:5]} "
        f"(showing up to 5 of {len(mismatches)})"
    )


def test_realistic_chunk_speedup_floor_on_long_program() -> None:
    """Wall-time benchmark: probing LONG_PROGRAM with realistic tokenizer-piece
    chunks through the suffix-anchor path must be meaningfully faster than
    the forced full-prefix-relex baseline. Threshold (1.2x) matches PR
    #1176's own floor -- kept well under the ~1.56x measured locally -- since
    per-call `InteractiveParser.copy()` overhead (unaffected by this fix,
    unaffected by chunk size) still dilutes the wall-time win relative to
    the larger chars_lexed reduction measured above."""
    chunks = _realistic_chunks()

    def _timed(fn, n=5):
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        samples.sort()
        return samples[len(samples) // 2]

    new_median = _timed(lambda: _drive_realistic_chunks(engine_mod.OpenUIIncrementalEngine(), chunks))

    orig = engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor
    engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = _disabled_anchor
    try:
        old_median = _timed(
            lambda: _drive_realistic_chunks(engine_mod.OpenUIIncrementalEngine(), chunks)
        )
    finally:
        engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = orig

    assert new_median * 1.2 < old_median, (
        f"wall-time speedup below the 1.2x floor at realistic chunk size: "
        f"new={new_median:.4f}s old={old_median:.4f}s"
    )
