"""Aggregate multi-candidate fan-out benefit of the suffix-anchor fixes
(docs/design/decode-compiler-tree-completion-forest-aggregate-benefit-finding.md).

PR #1177's realistic-chunk-size finding
(decode-compiler-tree-probe-chunk-realistic-chunk-size-finding.md) measured
`probe_chunk`'s suffix-anchor fix on a single straight-line candidate drive
and flagged, as its "Next steps" item 1, that `build_completion_forest`'s
real call site fans out into *many* short-lived `OpenUIIncrementalEngine`
branches per call (one per candidate token, most rejected within 1-2
probes; `pack.py`'s recursive `_tail_from` search multiplies this further),
and that the *aggregate* wall-clock/lex-work benefit across that fan-out
"is a different quantity" from the single-candidate figure and was
unmeasured.

This module closes that gap: it drives the real `build_completion_forest`
entry point (via `pack.py`'s `remaining_tokens`-bounded path, which is what
actually exercises the recursive multi-candidate search) on the same
PREFIX8/PREFIX9 fixtures PRs #1171/#1174/#1176 established, with both a
per-engine-construction counter and a `_lex_tokens` char counter installed,
comparing the real suffix-anchor fixes (#1174 `_incremental_sync`, #1176
`probe_chunk`) against a forced "before" oracle with both disabled.
"""

from __future__ import annotations

import time

import pytest

from slm_training.dsl.grammar.fastpath import engine as engine_mod
from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

# Exact prefixes from PR #1171's dead_end_trace, reused throughout this
# finding chain (test_grammar_fastpath_lexer_cache.py,
# test_grammar_fastpath_incremental_lex.py). PREFIX9's 12-candidate forest,
# combined with pack.py's recursive `_tail_from` search, is what turns "12
# top-level candidates" into thousands of short-lived branch engines.
PREFIX8 = "root = Card([b1"
PREFIX9 = "root = Card([b1,"


def _forced_full_relex_incremental_sync(self, prefix):
    """Self-contained "before" oracle for `_incremental_sync` -- duplicated
    from test_grammar_fastpath_incremental_lex.py per this chain's own
    convention of a small self-contained copy per fix/finding module rather
    than a cross-file import."""
    assert self._ip is not None
    tokens = self._lex_tokens(prefix)
    if tokens is None:
        return self._full_sync(prefix)
    keys = self._token_keys(tokens)
    if len(tokens) < self._fed_token_count or keys[: self._fed_token_count] != self._fed_tokens:
        return self._full_sync(prefix)
    return self._feed_delta_tokens(
        prefix, tokens[self._fed_token_count :], keys, base_offset=0
    )


def _disabled_anchor(self, chunk: str) -> None:
    """Self-contained "before" oracle for `probe_chunk`'s suffix-anchor
    path -- duplicated from test_grammar_fastpath_probe_chunk_realistic_chunk_size.py
    per the same convention."""
    return


class _Measurement:
    """Named result instead of a bare tuple -- keeps call sites self-documenting."""

    __slots__ = ("chars_lexed", "engines_created", "forest")

    def __init__(self, chars_lexed: int, engines_created: int, forest) -> None:
        self.chars_lexed = chars_lexed
        self.engines_created = engines_created
        self.forest = forest


def _measure(tok, prefix_ids: list[int]) -> _Measurement:
    """Run one real `build_completion_forest` call (`pack.py`'s
    `remaining_tokens`-bounded, recursively-searched path) with a per-engine
    construction counter and a `_lex_tokens` char counter installed, against
    whatever `_incremental_sync`/`_probe_delta_from_anchor` are bound to at
    call time -- the real fix, or a forced "before" oracle, depending on the
    caller."""
    counts = {"chars": 0, "engines": 0}
    orig_init = engine_mod.OpenUIIncrementalEngine.__init__
    orig_lex = engine_mod.OpenUIIncrementalEngine._lex_tokens

    def _counting_init(self, *a, **kw):
        counts["engines"] += 1
        return orig_init(self, *a, **kw)

    def _counting_lex(self, prefix):
        counts["chars"] += len(prefix)
        return orig_lex(self, prefix)

    engine_mod.OpenUIIncrementalEngine.__init__ = _counting_init
    engine_mod.OpenUIIncrementalEngine._lex_tokens = _counting_lex
    try:
        forest = build_completion_forest(
            tok, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
        )
    finally:
        engine_mod.OpenUIIncrementalEngine.__init__ = orig_init
        engine_mod.OpenUIIncrementalEngine._lex_tokens = orig_lex
    return _Measurement(counts["chars"], counts["engines"], forest)


def _measure_before(tok, prefix_ids: list[int]) -> _Measurement:
    """Same as `_measure`, with both suffix-anchor fixes forced off -- the
    pre-#1174/#1176 full-relex baseline."""
    orig_sync = engine_mod.OpenUIIncrementalEngine._incremental_sync
    orig_anchor = engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor
    engine_mod.OpenUIIncrementalEngine._incremental_sync = _forced_full_relex_incremental_sync
    engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = _disabled_anchor
    try:
        return _measure(tok, prefix_ids)
    finally:
        engine_mod.OpenUIIncrementalEngine._incremental_sync = orig_sync
        engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = orig_anchor


@pytest.fixture(scope="module")
def tokenizer():
    return DSLNativeTokenizer.build()


@pytest.fixture(scope="module", params=[PREFIX8, PREFIX9], ids=["prefix8", "prefix9"])
def prefix_ids(request, tokenizer):
    return [tokenizer.bos_id, *tokenizer.encode(request.param, add_special=False)]


@pytest.fixture(scope="module")
def aggregate_measurement(tokenizer, prefix_ids):
    return _measure(tokenizer, prefix_ids), _measure_before(tokenizer, prefix_ids)


def test_multi_candidate_fan_out_creates_many_branch_engines(aggregate_measurement) -> None:
    """Sanity check on this module's premise: the real `build_completion_forest`
    call site spawns many short-lived `OpenUIIncrementalEngine` branches per
    call, not the single straight-line candidate PR #1177's LONG_PROGRAM
    drive exercised."""
    after, _before = aggregate_measurement
    assert after.engines_created > 50, (
        f"expected build_completion_forest to fan out into many branch "
        f"engines, got {after.engines_created}"
    )


def test_build_completion_forest_result_unchanged_with_both_fixes_disabled(
    aggregate_measurement,
) -> None:
    """I3/I6: coverage/n_paths must be identical whether the suffix-anchor
    fixes are live or forced back to the pre-#1174/#1176 full-relex
    baseline, across the full multi-candidate fan-out -- not just the one
    straight-line candidate PRs #1174/#1176 already proved this for."""
    after, before = aggregate_measurement
    assert after.forest.coverage == before.forest.coverage == "complete"
    assert len(after.forest.paths) == len(before.forest.paths)


def test_aggregate_chars_lexed_reduced_but_far_below_single_candidate_ratio(
    aggregate_measurement,
) -> None:
    """Regression guard + the actual "next step 1" measurement: aggregated
    across the real multi-candidate fan-out, the suffix-anchor fixes still
    reduce total chars lexed -- but the aggregate ratio lands far below PR
    #1177's 7.29x single-candidate figure, because most branches are
    short-lived (a handful of probes before rejection) and start
    anchor-less, leaving little room for the suffix-anchor path to
    compound."""
    after, before = aggregate_measurement
    assert after.chars_lexed < before.chars_lexed, (
        f"suffix-anchor fixes did not reduce aggregate chars_lexed: "
        f"after={after.chars_lexed} before={before.chars_lexed}"
    )
    # Conservative floor, kept well under the ~2.4x measured locally on both
    # PREFIX8 and PREFIX9 (see the finding doc).
    assert after.chars_lexed * 1.8 < before.chars_lexed, (
        f"aggregate chars_lexed reduction below the 1.8x floor: "
        f"after={after.chars_lexed} before={before.chars_lexed}"
    )
    # And explicitly far below PR #1177's 7.29x single-candidate-drive
    # figure -- proof this finding's premise (short-lived branches shrink
    # the aggregate win) actually holds, not just that it's unmeasured.
    assert after.chars_lexed * 5.0 > before.chars_lexed, (
        "aggregate chars_lexed reduction unexpectedly matched or exceeded "
        "PR #1177's 7.29x single-candidate figure -- re-check whether this "
        "finding's premise still holds: "
        f"after={after.chars_lexed} before={before.chars_lexed}"
    )


def test_aggregate_wall_time_speedup_floor_on_prefix9(tokenizer) -> None:
    """Wall-time benchmark on PREFIX9 specifically (the richer, 12-candidate,
    thousands-of-branch-engine fan-out) -- median of 3 reps each side. Floor
    (1.05x) is well under the ~1.15-1.2x measured locally, reflecting this
    module's own finding that the aggregate wall-time win is real but much
    smaller than PR #1177's 1.56x single-candidate figure: most branches are
    too short-lived for `InteractiveParser.copy()`'s fixed per-call cost to
    be diluted by a large lex-work reduction."""
    prefix_ids = [tokenizer.bos_id, *tokenizer.encode(PREFIX9, add_special=False)]

    def _timed(fn, n=3):
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        samples.sort()
        return samples[len(samples) // 2]

    after_median = _timed(
        lambda: build_completion_forest(
            tokenizer, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
        )
    )

    orig_sync = engine_mod.OpenUIIncrementalEngine._incremental_sync
    orig_anchor = engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor
    engine_mod.OpenUIIncrementalEngine._incremental_sync = _forced_full_relex_incremental_sync
    engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = _disabled_anchor
    try:
        before_median = _timed(
            lambda: build_completion_forest(
                tokenizer, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
            )
        )
    finally:
        engine_mod.OpenUIIncrementalEngine._incremental_sync = orig_sync
        engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = orig_anchor

    assert after_median * 1.05 < before_median, (
        f"aggregate wall-time speedup below the 1.05x floor on PREFIX9: "
        f"after={after_median:.4f}s before={before_median:.4f}s"
    )
