"""Regression tests for the cached grammar-path resolver in
``slm_training.dsl.grammar.fastpath.engine`` (docs/design/
decode-compiler-tree-witness-search-cost-finding.md).

That finding's cProfile showed ``OpenUIIncrementalEngine.__init__`` calling
``path.resolve()`` twice per construction (once for ``_load_parser``, once
for ``_load_lexer``) -- 10,967 inits in one profiled decode record, ~2.9s +
~2.2s cumulative ``Path.resolve``/``os.path.realpath`` cost -- even though
the raw grammar path is one of a handful of constants for a process's
lifetime. ``_resolve_grammar_path`` (``@lru_cache``) now pays that syscall
once per distinct raw path string; ``__init__`` calls it once and reuses the
result for both ``_load_parser`` and ``_load_lexer``.

These tests prove: (a) ``Path.resolve()`` is called at most once across many
engine constructions, (b) equivalent-but-differently-spelled raw paths still
share the same cached ``Lark``/``BasicLexer`` objects (the resolver's cache
key is the raw string, but its value is the one canonical resolved path, so
``_load_parser``/``_load_lexer``'s own cache -- keyed on that resolved
string -- is unaffected), and (c) decode semantics are unchanged.
"""

from __future__ import annotations

from pathlib import Path

from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
from slm_training.dsl.grammar.fastpath import engine as engine_mod
from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest

# Same finding prefix as test_grammar_fastpath_lexer_cache.py's PREFIX9.
PREFIX9 = "root = Card([b1,"


def test_resolve_grammar_path_is_cached() -> None:
    engine_mod._resolve_grammar_path.cache_clear()
    raw = str(GRAMMARS_DIR / "openui.lark")
    first = engine_mod._resolve_grammar_path(raw)
    second = engine_mod._resolve_grammar_path(raw)
    assert first == second == str((GRAMMARS_DIR / "openui.lark").resolve())


def test_path_resolve_called_at_most_once_across_many_inits(monkeypatch) -> None:
    """Call-count regression guard: the finding measured 10,967 uncached
    ``path.resolve()`` calls (2 per init) for one decode-time record. After
    the fix, ``Path.resolve`` runs at most once no matter how many engines
    are constructed for the same grammar path."""
    engine_mod._resolve_grammar_path.cache_clear()
    engine_mod._load_parser.cache_clear()
    engine_mod._load_lexer.cache_clear()

    resolve_calls = {"n": 0}
    orig_resolve = Path.resolve

    def counting_resolve(self, *args, **kwargs):
        resolve_calls["n"] += 1
        return orig_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counting_resolve)

    for _ in range(25):
        engine_mod.OpenUIIncrementalEngine()

    assert resolve_calls["n"] <= 1


def test_equivalent_raw_paths_share_the_same_cached_parser_and_lexer() -> None:
    """Two different raw spellings of the same grammar file (the default
    unresolved ``Path`` vs an explicit, already-resolved one) must still
    hit the same ``_load_parser``/``_load_lexer`` cache entries."""
    engine_mod._resolve_grammar_path.cache_clear()
    engine_mod._load_parser.cache_clear()
    engine_mod._load_lexer.cache_clear()

    default_engine = engine_mod.OpenUIIncrementalEngine()
    explicit_engine = engine_mod.OpenUIIncrementalEngine(
        (GRAMMARS_DIR / "openui.lark").resolve()
    )
    assert default_engine._parser is explicit_engine._parser
    assert default_engine._lexer is explicit_engine._lexer


def test_build_completion_forest_unchanged_after_path_cache_fix() -> None:
    """Decode-invariant proof (I3/I6): a pure path-resolution caching change
    must never alter ``coverage``/``n_paths`` (same prefix/expectation as
    test_grammar_fastpath_lexer_cache.py's PR #1173 regression test)."""
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    prefix_ids = [tok.bos_id, *tok.encode(PREFIX9, add_special=False)]
    forest = build_completion_forest(
        tok, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
    )
    assert forest.coverage == "complete"
    assert len(forest.paths) == 12
