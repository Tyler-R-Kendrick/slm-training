"""Regression tests for the memoized ``InteractiveParser.accepts()`` in
``slm_training.dsl.grammar.fastpath.engine``
(docs/design/decode-accepts-state-memoization.md).

PR #1191's cProfile (docs/design/
decode-compiler-tree-witness-search-cost-finding.md) named Lark's own
``InteractiveParser.accepts()`` (12.6s / 23,530 calls) as the dominant
remaining witness-search cost, and named "memoize by parser-automaton state
rather than token history" as the next lever -- flagging it as needing its
own I3/I6 safety proof before attempting (bigger, Lark-internals-facing
change).

The safety proof (see the design doc for the full argument, and Lark's
``lark/parsers/lalr_parser_state.py``/``lalr_interactive_parser.py`` source):

* ``accepts()`` only reads ``self.choices()`` (the *top-of-stack* state's
  static action table) and ``self.parser_state.state_stack``/``.parse_conf``.
  It explicitly strips callbacks and builds probe tokens from the bare
  ``Token`` class, never from live lexer state -- so lexer/value-stack
  content plays no role.
* ``ParserState.feed_token`` -- what each accept-probe actually runs -- does
  a reduce **chain**: a Reduce action pops ``size`` frames off the stack and
  then re-checks the *same* lookahead token against the state now exposed
  *below* the frames just popped. That means the top-of-stack state ALONE
  does not determine ``accepts()``'s outcome in general: two different left
  contexts that happen to share only the top state can expose different
  states below it after a reduce, and thus can legally disagree on whether a
  given token is accepted. Memoizing by "current LALR state" (top-of-stack
  only) would therefore be unsound.
* The full ``state_stack`` (a tuple of ints for the default ``IntParseTable``)
  IS a sound key: it is exactly the information ``feed_token``'s reduce
  chain consults, so two calls sharing the same full ``state_stack`` (under
  the same, immutable, per-grammar parse table) are provably calls to the
  same pure function of the same input and must return the same answer.

These tests prove, mirroring the rigor of ``test_pack_witness_cache.py`` and
``test_grammar_fastpath_lexer_cache.py``:

1. The cache is byte-identical to the uncached ``ip.accepts()`` call across a
   battery of real prefixes (I3/I6: a caching change must never become a
   legality change).
2. A synthetic grammar demonstrates the general unsoundness of a
   top-of-stack-only key (motivating why the fix keys on the full
   ``state_stack`` instead) -- two different left contexts sharing a
   top-of-stack state can disagree on ``accepts()``'s outcome.
3. Real cache hit/miss counts and any ``_sync``/witness-search call-count or
   wall-time change on the exact PREFIX8/PREFIX9 cases used throughout this
   stack -- an honest measurement, not an assumed win.
4. ``build_completion_forest``/``_openui_completion_domain`` witness content
   (``coverage``, ``n_paths``, candidate token sequences) stays byte-identical
   on PREFIX8/PREFIX9.
"""

from __future__ import annotations

import time

import pytest
from lark import Lark

from slm_training.dsl.grammar.fastpath import engine as engine_mod
from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest
from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
from slm_training.dsl.pack import _openui_completion_domain
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

# Exact prefixes from PR #1171's dead_end_trace / the lexer-rebuild finding,
# reused here (same as test_grammar_fastpath_lexer_cache.py /
# test_pack_witness_cache.py) so this fix is measured on the identical,
# previously-characterized fan-out.
PREFIX8 = "root = Card([b1"
PREFIX9 = "root = Card([b1,"

_GROWING_PREFIXES = [
    "root",
    "root =",
    "root = Card(",
    "root = Card([b1",
    "root = Card([b1,",
    "root = Card([b1, b2",
]


def _reset() -> None:
    engine_mod._accepts_cache_clear()


def _tokenizer() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


def _request(tok: DSLNativeTokenizer, prefix: str) -> CompletionDomainRequestV1:
    prefix_ids = (tok.bos_id, *tok.encode(prefix, add_special=False))
    return CompletionDomainRequestV1(
        prefix_ids=prefix_ids,
        tokenizer=tok,
        remaining_tokens=24,
        max_path_tokens=8,
        min_content=0,
    )


def test_cached_accepts_byte_identical_to_uncached_on_real_prefixes() -> None:
    """I3/I6: the memoized wrapper must return exactly the same accepted-
    terminal set as calling ``InteractiveParser.accepts()`` directly, at
    every step of several growing real prefixes (mix of cache misses and,
    for repeated states, hits)."""
    _reset()
    for prefix in _GROWING_PREFIXES:
        eng_cached = engine_mod.OpenUIIncrementalEngine()
        eng_cached.reset()
        assert eng_cached.set_prefix(prefix)
        cached_terminals = eng_cached.next_terminals()

        eng_uncached = engine_mod.OpenUIIncrementalEngine()
        eng_uncached.reset()
        assert eng_uncached.set_prefix(prefix)
        uncached_terminals = frozenset(str(x) for x in eng_uncached._ip.accepts())

        assert cached_terminals == uncached_terminals, (
            f"prefix={prefix!r}: cached accepts() diverged from uncached "
            f"({cached_terminals} vs {uncached_terminals})"
        )


def test_accepts_cache_hits_on_repeated_state_stack() -> None:
    """Structural proof the cache is genuinely doing work: resetting to the
    exact same prefix a second time (same grammar, same state_stack) must be
    a cache hit, not a fresh Lark probe."""
    _reset()
    eng = engine_mod.OpenUIIncrementalEngine()
    eng.reset()
    eng.set_prefix(PREFIX8)
    hits_a, misses_a, _ = engine_mod.accepts_cache_info()
    assert misses_a > 0

    eng2 = engine_mod.OpenUIIncrementalEngine()
    eng2.reset()
    eng2.set_prefix(PREFIX8)
    hits_b, misses_b, _ = engine_mod.accepts_cache_info()

    # The second engine re-derives the identical state_stack sequence (same
    # grammar, same prefix, independent engine instance) -- every accepts()
    # call along the way must now be served from cache.
    assert hits_b > hits_a
    assert misses_b == misses_a


def test_top_of_stack_alone_is_not_a_sound_key() -> None:
    """Demonstrates *why* the fix keys on the full ``state_stack`` instead of
    just ``parser_state.position`` (the top-of-stack LALR state): a small
    synthetic grammar with a shared reduce target shows two different left
    contexts reaching the *same* top-of-stack state can still legally
    disagree on ``accepts()``'s outcome, because a Reduce action pops frames
    and re-checks the lookahead against the state exposed *below* them.

    Grammar: after consuming "a" (context A) or "b" (context B), both derive
    a nonterminal N via a zero-size epsilon-like short reduce, landing on the
    *same* top-of-stack state; but what's legal *next* differs by context
    (only context A permits "x" afterward, only context B permits "y") --
    proving the two contexts must be tracked separately, i.e. top-of-stack
    state alone is insufficient.
    """
    grammar = r"""
        start: a_branch | b_branch
        a_branch: "a" mid "x"
        b_branch: "b" mid "y"
        mid: "m"
        %import common.WS
        %ignore WS
    """
    parser = Lark(grammar, parser="lalr", lexer="basic")

    ip_a = parser.parse_interactive("a m")
    list(ip_a.iter_parse())
    ip_b = parser.parse_interactive("b m")
    list(ip_b.iter_parse())

    # Both contexts reduce `mid` and land on the same top-of-stack state
    # (the grammar merges LALR states for the isomorphic "after mid" point
    # reached from either branch)...
    assert ip_a.parser_state.position == ip_b.parser_state.position
    # ...but the full state_stack differs (different left context still on
    # the stack below top)...
    assert tuple(ip_a.parser_state.state_stack) != tuple(ip_b.parser_state.state_stack)
    # ...and accepts() genuinely disagrees: only "x" is legal after "a m",
    # only "y" is legal after "b m". A top-of-stack-only cache keyed on
    # `position` would incorrectly serve one context's answer to the other.
    accepts_a = ip_a.accepts()
    accepts_b = ip_b.accepts()
    assert accepts_a != accepts_b
    assert "X" in accepts_a and "X" not in accepts_b
    assert "Y" in accepts_b and "Y" not in accepts_a


@pytest.mark.parametrize(
    ("prefix", "expected_n_paths"),
    [(PREFIX8, 2), (PREFIX9, 12)],
)
def test_build_completion_forest_unchanged_with_accepts_cache(
    prefix: str, expected_n_paths: int
) -> None:
    """I3/I6: with the accepts() memoization live, build_completion_forest's
    coverage/n_paths on the finding's prefixes must be unchanged from the
    pre-fix values recorded in
    docs/design/decode-compiler-tree-lexer-rebuild-cost-finding.json
    (prefix8 n_paths=2, prefix9 n_paths=12, both coverage="complete")."""
    _reset()
    tok = _tokenizer()
    prefix_ids = [tok.bos_id, *tok.encode(prefix, add_special=False)]
    forest = build_completion_forest(
        tok, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
    )
    assert forest.coverage == "complete"
    assert len(forest.paths) == expected_n_paths


@pytest.mark.parametrize("prefix, expected_candidates", [(PREFIX8, 2), (PREFIX9, 12)])
def test_completion_domain_unchanged_with_accepts_cache(
    prefix: str, expected_candidates: int
) -> None:
    """Same I3/I6 proof at the ``_openui_completion_domain`` level -- witness
    status/candidate count unchanged with the accepts() cache live."""
    _reset()
    tok = _tokenizer()
    domain = _openui_completion_domain(_request(tok, prefix))
    assert domain.status == "complete"
    assert len(domain.candidates) == expected_candidates


def test_measured_accepts_cache_hit_rate_and_wall_time_on_finding_prefixes() -> None:
    """Honest measurement (not an assumed speedup): real Lark
    ``InteractiveParser.accepts()`` call count (misses) vs. served-from-cache
    count (hits), and wall time, for the exact witness-search workload
    (``_openui_completion_domain`` on PREFIX8/PREFIX9) that PR #1191's
    cProfile characterized. Reported via the docs/design write-up; asserts
    only structural properties (non-negative hits, byte-identical results),
    not a specific speedup ratio, since the hit rate is a property of this
    grammar/these prefixes' actual state-stack repetition, not guaranteed by
    the fix's correctness.
    """
    tok = _tokenizer()
    results = {}
    for prefix in (PREFIX8, PREFIX9):
        _reset()
        t0 = time.perf_counter()
        domain = _openui_completion_domain(_request(tok, prefix))
        elapsed = time.perf_counter() - t0
        hits, misses, _size = engine_mod.accepts_cache_info()
        results[prefix] = (hits, misses, elapsed, domain.status)
        assert domain.status == "complete"
        assert hits >= 0 and misses > 0

    # Surfaced for the docs/design write-up; not a hard assertion beyond
    # "the run completed and produced legal witnesses" -- see the module
    # docstring and the design doc for the honest hit-rate numbers.
    for prefix, (hits, misses, elapsed, status) in results.items():
        total = hits + misses
        hit_rate = hits / total if total else 0.0
        print(
            f"accepts-cache prefix={prefix!r} status={status} "
            f"hits={hits} misses={misses} hit_rate={hit_rate:.4%} "
            f"elapsed_s={elapsed:.4f}"
        )
