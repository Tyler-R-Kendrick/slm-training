"""Regression tests for the hoisted ``_tail`` witness-search cache in
``slm_training.dsl.pack._openui_completion_domain``
(docs/design/decode-compiler-tree-witness-search-cache-hoist.md).

The predecessor finding
(docs/design/decode-compiler-tree-witness-search-cost-finding.md) showed
``_openui_completion_domain`` created a brand-new ``@lru_cache``-wrapped
``_tail`` closure (pack.py:402) *inside* a fresh ``_tail_from`` call for
*every* top-level candidate in ``initial.paths``, so sibling candidates could
never reuse each other's proven witness-search states. The fix hoists the
``lru_cache``-wrapped ``_tail`` (and its ``nodes_left`` fairness budget box)
out of the per-candidate loop so it is built exactly once per
``_openui_completion_domain`` call.

These tests prove three things, matching the rigor of
``tests/test_dsl/test_grammar_fastpath_lexer_cache.py``'s lexer-cache fix:

1. The cache is genuinely created once per call, not once per candidate
   (structural proof the hoist happened) -- this test fails against the
   pre-fix code (manually confirmed via ``git stash`` during development:
   pre-fix ``root = Card([b1`` creates 2 ``_tail`` caches and
   ``root = Card([b1,`` creates 14, one per non-eos top-level candidate).
2. Witness *content* (which completions are found, and their exact token
   sequences) and ``coverage``/candidate count are byte-identical before and
   after (I3/I6: a caching-scope change must never become a legality
   change).
3. ``nodes_left`` fairness (each top-level candidate gets its own fresh 16
   *novel*-state chances) is preserved exactly, even though the underlying
   ``lru_cache`` is now shared -- a synthetic always-novel grammar stub
   proves per-candidate call counts are unchanged by the hoist.

Honest result (see the finding doc): hoisting is safe and reduces cache
*object* churn, but does **not** reduce ``_sync``/forest-build call counts on
either measured prefix, because the cache key (`current` token-id history +
remaining room) never collides -- 0 hits out of thousands of misses, on both
the pre-fix and post-fix code. `test_sync_call_count_unchanged_by_hoist`
documents that honestly instead of claiming a speedup that was not
measured.
"""

from __future__ import annotations

import functools

import pytest

from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
from slm_training.dsl.grammar.fastpath import compiler_draft as compiler_draft_mod
from slm_training.dsl.grammar.fastpath import engine as engine_mod
from slm_training.dsl.grammar.fastpath.compiler_draft import CompletionForest, CompletionPath
from slm_training.dsl.pack import _openui_completion_domain
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

# Exact prefixes from PR #1171's dead_end_trace / the lexer-rebuild finding,
# reused here (same as test_grammar_fastpath_lexer_cache.py) so the two
# fixes are measured on the identical, previously-characterized fan-out.
PREFIX8 = "root = Card([b1"
PREFIX9 = "root = Card([b1,"

# Golden candidate content captured directly from `_openui_completion_domain`
# both before (git-stashed HEAD, i.e. the un-hoisted per-candidate cache) and
# after this fix -- byte-identical in both runs, confirming the hoist cannot
# change witness content (see module docstring point 2).
_GOLDEN_PREFIX8 = {
    "status": "complete",
    "reason": "",
    "candidates": [
        ("grammar_rsqb_root_populated", (9,), (9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        (
            "grammar_comma",
            (12,),
            (12, 33, 6, 8, 32, 6, 85, 7, 9, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2),
        ),
    ],
}
_GOLDEN_PREFIX9 = {
    "status": "complete",
    "reason": "witness_pruned",
    "candidates": [
        ("component", (33, 6), (33, 6, 8, 32, 6, 85, 7, 9, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        ("component", (34, 6), (34, 6, 167, 12, 85, 12, 85, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        ("component", (36, 6), (36, 6, 85, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        ("component", (43, 6), (43, 6, 85, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        ("component", (44, 6), (44, 6, 85, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        ("component", (45, 6), (45, 6, 8, 9, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        ("component", (55, 6), (55, 6, 171, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        ("component", (57, 6), (57, 6, 8, 32, 6, 85, 7, 9, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        ("component", (62, 6), (62, 6, 8, 378, 9, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        ("component", (64, 6), (64, 6, 160, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        ("component", (65, 6), (65, 6, 85, 7, 9, 7, 30, 378, 5, 32, 6, 85, 7, 2)),
        (
            "bind_reference_root_children",
            (379,),
            (379, 9, 7, 30, 378, 5, 32, 6, 85, 7, 30, 379, 5, 32, 6, 85, 7, 2),
        ),
    ],
}

# Measured real-grammar `_sync` call counts on the exact same prefixes,
# unchanged by the hoist (see module docstring / point in finding doc: the
# `_tail` cache never records a hit, before or after, so there is nothing
# for a wider cache scope to save on these prefixes).
_EXPECTED_SYNC_CALLS = {PREFIX8: 1073, PREFIX9: 16082}


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


class _LruCacheSpy:
    """Counts `@lru_cache`-wrapped functions created while active, filtered
    by qualname so it is robust to whatever else got lru_cache-decorated on
    first import in this process (`_load_parser`, `_load_lexer`, stdlib
    `typing` internals, ...) -- only `_tail` closures from
    `_openui_completion_domain` are counted."""

    def __init__(self) -> None:
        self.tail_caches: list[object] = []
        self._orig = functools.lru_cache

    def __enter__(self) -> "_LruCacheSpy":
        orig = self._orig

        def spy(*args, **kwargs):
            decorator = orig(*args, **kwargs)

            def wrap(fn):
                wrapped = decorator(fn)
                qualname = getattr(getattr(wrapped, "__wrapped__", None), "__qualname__", "")
                if "_openui_completion_domain" in qualname and qualname.endswith("._tail"):
                    self.tail_caches.append(wrapped)
                return wrapped

            return wrap

        functools.lru_cache = spy
        return self

    def __exit__(self, *exc: object) -> None:
        functools.lru_cache = self._orig


@pytest.mark.parametrize("prefix, expected_candidates", [(PREFIX8, 2), (PREFIX9, 12)])
def test_tail_cache_created_once_per_completion_domain_call(
    prefix: str, expected_candidates: int
) -> None:
    """Structural proof the hoist happened: exactly one `_tail` `lru_cache`
    is built for the whole call, regardless of how many top-level candidates
    it serves. Pre-fix, this was one `lru_cache` per non-eos top-level
    candidate (2 for PREFIX8, 14 for PREFIX9 -- confirmed by `git stash`ing
    this fix during development and rerunning this exact probe)."""
    tok = _tokenizer()
    with _LruCacheSpy() as spy:
        domain = _openui_completion_domain(_request(tok, prefix))
    assert len(spy.tail_caches) == 1
    assert len(domain.candidates) == expected_candidates


@pytest.mark.parametrize(
    "prefix, golden", [(PREFIX8, _GOLDEN_PREFIX8), (PREFIX9, _GOLDEN_PREFIX9)]
)
def test_completion_domain_candidates_byte_identical_to_pre_hoist(
    prefix: str, golden: dict
) -> None:
    """I3/I6: witness content (kind, token_ids, terminal_witness for every
    candidate) and status/reason must be byte-identical to the pre-hoist
    values captured from HEAD before this fix."""
    tok = _tokenizer()
    domain = _openui_completion_domain(_request(tok, prefix))
    assert domain.status == golden["status"]
    assert domain.reason == golden["reason"]
    observed = [
        (candidate.kind, tuple(candidate.token_ids), tuple(candidate.terminal_witness))
        for candidate in domain.candidates
    ]
    assert observed == golden["candidates"]


def test_sync_call_count_unchanged_by_hoist() -> None:
    """Honest benchmark (not a speedup claim): `OpenUIIncrementalEngine._sync`
    call count on PREFIX8/PREFIX9 is unchanged by the hoist. The predecessor
    finding hypothesized sibling candidates re-explore identical grammar
    states; measured directly, the shared `_tail` cache's `cache_info()`
    shows 0 hits (all misses) on both prefixes -- the `(current, room)` key
    is an absolute token-id history that never collides across candidates OR
    within one candidate's own recursion (both `current` and `room` change
    monotonically along every recursive call), so there was never any
    redundant computation for a wider cache scope to eliminate."""
    tok = _tokenizer()
    for prefix, expected in _EXPECTED_SYNC_CALLS.items():
        calls = {"n": 0}
        orig_sync = engine_mod.OpenUIIncrementalEngine._sync

        def counting_sync(self, *args, orig=orig_sync, **kwargs):
            calls["n"] += 1
            return orig(self, *args, **kwargs)

        engine_mod.OpenUIIncrementalEngine._sync = counting_sync
        try:
            domain = _openui_completion_domain(_request(tok, prefix))
        finally:
            engine_mod.OpenUIIncrementalEngine._sync = orig_sync
        assert domain.status == "complete"
        assert calls["n"] == expected, (
            f"{prefix!r}: _sync call count changed ({calls['n']} vs "
            f"{expected} baseline) -- the hoist is meant to be a no-op on "
            "call count for these prefixes; a change here means the cache "
            "key semantics shifted and needs re-investigation, not just a "
            "docstring update"
        )


class _FakeTokenizer:
    """Minimal stand-in with a callable `kind_ids` so `_openui_completion_domain`
    takes the engine-native fast path (not the word-tokenizer projection
    fallback at pack.py:307-371)."""

    bos_id = 0
    eos_id = 1

    @staticmethod
    def kind_ids(*_args: object, **_kwargs: object) -> None:
        return None


def _always_novel_forest(tokenizer: object, current: list[int], **_kwargs: object) -> CompletionForest:
    """A synthetic grammar stub: every prefix has exactly one legal,
    never-EOS continuation of a single fresh token, so recursion can only
    stop when `nodes_left` (the fairness budget) or `room` is exhausted --
    never because a real EOS was found. Used to prove `nodes_left` fairness
    is unaffected by cache hoisting."""
    next_token = 9000 + len(current)
    return CompletionForest(paths=(CompletionPath(token_ids=(next_token,), kind="filler"),), coverage="complete")


def test_nodes_left_fairness_unchanged_by_hoist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fairness proof: with a synthetic grammar where no state is ever
    revisited (every `current` is strictly longer than the last, so the
    shared cache can never produce a hit -- by construction, not by luck),
    each of N top-level candidates must still get its own fresh 16-node
    budget. If the hoist had accidentally shared `nodes_left` itself (not
    just the cache), the *first* candidate would exhaust the budget and
    every later candidate would get 0 real chances -- so total real
    forest-build calls would plateau at ``1 + 16`` instead of scaling with N.
    """
    monkeypatch.setattr(
        compiler_draft_mod, "_build_openui_completion_forest", _always_novel_forest
    )
    build_calls = {"n": 0}
    real_fake = _always_novel_forest

    def counting_fake(*args: object, **kwargs: object) -> CompletionForest:
        build_calls["n"] += 1
        return real_fake(*args, **kwargs)

    monkeypatch.setattr(compiler_draft_mod, "_build_openui_completion_forest", counting_fake)

    # `initial.paths` is the top-level candidate list `_openui_completion_domain`
    # loops over; give it 5 non-eos candidates via the same synthetic stub
    # (`_always_novel_forest`/`counting_fake` handles both the initial call
    # and every recursive `_tail` call identically).
    n_top_level = 5
    tok = _FakeTokenizer()
    request = CompletionDomainRequestV1(
        prefix_ids=(0,),
        tokenizer=tok,
        remaining_tokens=1000,
        max_path_tokens=8,
        min_content=0,
    )

    # The initial call returns only one path from `_always_novel_forest`;
    # patch it once more, just for the *first* call, to expose `n_top_level`
    # distinct top-level candidates instead of one, so the loop in
    # `_openui_completion_domain` actually iterates N times.
    initial_paths = tuple(
        CompletionPath(token_ids=(100 + i,), kind="filler") for i in range(n_top_level)
    )
    calls_seen = {"n": 0}

    def counting_fake_with_fan_out(tokenizer: object, current: list[int], **kwargs: object):
        calls_seen["n"] += 1
        if calls_seen["n"] == 1:
            return CompletionForest(paths=initial_paths, coverage="complete")
        return real_fake(tokenizer, current, **kwargs)

    monkeypatch.setattr(
        compiler_draft_mod, "_build_openui_completion_forest", counting_fake_with_fan_out
    )

    domain = _openui_completion_domain(request)

    # 1 initial call + n_top_level candidates * 16 fresh nodes_left chances
    # each (every state is novel by construction, so every chance is a real
    # `_build_openui_completion_forest` call, not a cache hit).
    assert calls_seen["n"] == 1 + n_top_level * 16
    # None of the synthetic candidates ever reaches EOS, so every one is
    # unwitnessed -- confirms the loop really visited all N candidates
    # rather than stopping early once a shared budget ran out.
    assert domain.reason == "terminal_witness_unavailable"
    assert domain.status == "incomplete"
