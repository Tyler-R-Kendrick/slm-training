"""Regression tests for the cached grammar-path resolution in
``slm_training.dsl.grammar.fastpath.engine`` (docs/design/
decode-engine-init-resolve-cache.md).

PR #1191's cProfile (docs/design/
decode-compiler-tree-witness-search-cost-finding.md) showed
``OpenUIIncrementalEngine.__init__`` calling ``path.resolve()`` twice per
construction -- once for ``_load_parser``, once for ``_load_lexer`` -- and
that this happened *before* those functions' own ``@lru_cache`` keys were
computed, so the resolve/``os.path.realpath`` syscall ran on every single
engine construction (10,967 calls / ~5.1s combined in that record's decode)
even though the parser/lexer objects themselves were already being reused
from cache.

The fix adds ``_resolve_grammar_path``, an ``@lru_cache``-wrapped function
keyed on the raw (pre-resolve) ``grammar_path`` string, and has
``__init__`` call it once and reuse the result for both ``_load_parser`` and
``_load_lexer`` lookups.
"""

from __future__ import annotations

import pathlib

from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
from slm_training.dsl.grammar.fastpath import engine as engine_mod


def _openui_path() -> pathlib.Path:
    return GRAMMARS_DIR / "openui.lark"


def _toy_layout_path() -> pathlib.Path:
    return GRAMMARS_DIR / "toy_layout.lark"


def _reset_caches() -> None:
    engine_mod._load_parser.cache_clear()
    engine_mod._load_lexer.cache_clear()
    engine_mod._resolve_grammar_path.cache_clear()


def test_resolve_call_count_drops_after_caching() -> None:
    """Call-count regression guard, matching the finding's cProfile: 200
    constructions of the same grammar path must trigger at most a handful of
    real ``Path.resolve()`` syscalls, not one (or two) per construction.

    Sanity-checked to fail meaningfully pre-fix: reverting the
    ``_resolve_grammar_path`` cache (equivalent to calling
    ``path.resolve()`` directly inside ``__init__``, as HEAD~1 does)
    reproduces exactly 2 real ``Path.resolve()`` calls per construction --
    400 for 200 constructions -- confirmed via ``git stash`` during
    development of this fix.
    """
    _reset_caches()
    calls = {"n": 0}
    orig_resolve = pathlib.Path.resolve

    def counting_resolve(self: pathlib.Path, *a: object, **kw: object) -> pathlib.Path:
        calls["n"] += 1
        return orig_resolve(self, *a, **kw)

    pathlib.Path.resolve = counting_resolve  # type: ignore[method-assign]
    try:
        for _ in range(200):
            engine_mod.OpenUIIncrementalEngine()
    finally:
        pathlib.Path.resolve = orig_resolve  # type: ignore[method-assign]

    # Exactly one real resolve for the one distinct grammar path used above
    # (the cache is process-lifetime, so a prior test in the same session
    # may have already warmed it -- assert "at most a few", not "==1", to
    # stay robust to test order while still catching the O(n) regression).
    assert calls["n"] <= 2, (
        f"expected Path.resolve() to be called at most twice (cache warm-up) "
        f"for 200 constructions of the same grammar path, got {calls['n']} -- "
        "the per-construction resolve cache regressed"
    )


def test_resolved_path_value_matches_uncached_resolve() -> None:
    """Correctness: the cached resolution must equal the value
    ``Path(grammar_path).resolve()`` would produce directly (I6 -- caching
    must never change the *value*, only how often it is computed)."""
    _reset_caches()
    raw = str(_openui_path())
    cached = engine_mod._resolve_grammar_path(raw)
    uncached = str(pathlib.Path(raw).resolve())
    assert cached == uncached

    eng = engine_mod.OpenUIIncrementalEngine(_openui_path())
    assert str(eng.grammar_path.resolve()) == cached


def test_resolve_cache_distinguishes_different_grammar_paths() -> None:
    """The cache must not collapse two different raw ``grammar_path``
    strings to the same resolved entry -- each distinct path gets its own
    cache slot and its own correct resolution."""
    _reset_caches()
    openui_raw = str(_openui_path())
    toy_raw = str(_toy_layout_path())
    assert openui_raw != toy_raw

    openui_resolved = engine_mod._resolve_grammar_path(openui_raw)
    toy_resolved = engine_mod._resolve_grammar_path(toy_raw)

    assert openui_resolved != toy_resolved
    assert openui_resolved == str(pathlib.Path(openui_raw).resolve())
    assert toy_resolved == str(pathlib.Path(toy_raw).resolve())

    # cache_info confirms both are distinct entries, not one shared hit.
    info = engine_mod._resolve_grammar_path.cache_info()
    assert info.currsize >= 2

    # Constructing engines for both paths must load the matching (distinct)
    # parser/lexer objects -- proves the resolve cache feeds the right key
    # into the downstream _load_parser/_load_lexer caches, not a collapsed
    # one-size-fits-all string.
    eng_openui = engine_mod.OpenUIIncrementalEngine(_openui_path())
    eng_toy = engine_mod.OpenUIIncrementalEngine(_toy_layout_path())
    assert eng_openui._parser is not eng_toy._parser
    assert eng_openui._lexer is not eng_toy._lexer


def test_engine_still_functions_after_resolve_caching() -> None:
    """End-to-end smoke: the engine built via the cached-resolve path still
    parses/accepts normally (I3 -- a caching change must never become a
    legality change)."""
    _reset_caches()
    eng = engine_mod.OpenUIIncrementalEngine()
    eng.reset()
    assert eng.set_prefix("root = Card([b1")
    assert eng.next_terminals()
