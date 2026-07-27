"""Regression tests for the cached ``BasicLexer`` in
``slm_training.dsl.grammar.fastpath.engine`` (docs/design/
decode-compiler-tree-lexer-rebuild-cost-finding.md).

That finding showed ``OpenUIIncrementalEngine._lex_tokens`` called
``self._parser.lex(prefix)`` -- Lark's standalone ``.lex()`` convenience
method -- on every ``_sync`` call. Because the engine is built with
``parser="lalr"``, ``Lark`` never caches a ``self.lexer`` attribute for that
path, so every ``.lex()`` call rebuilt a fresh ``BasicLexer`` (and lazily, a
fresh ``Scanner`` via ``_build_scanner`` / ``_create_unless``) from scratch --
23,546 rebuilds for a single ``build_completion_forest`` call at the
``prefix9`` case below (cProfile evidence in the finding doc).

The fix caches one ``BasicLexer`` per grammar path (``_load_lexer``,
alongside the existing ``_load_parser`` cache) and has ``_lex_tokens`` reuse
it via ``LexerThread``. These tests prove, on the exact prefixes from the
finding: (a) the scanner is built at most once regardless of call count,
(b) repeated lexing is measurably faster once cached, and (c) the *decode
semantics* -- the raw token stream and ``build_completion_forest``'s
``coverage``/``n_paths`` -- are byte-identical to the pre-fix path (I3/I6:
a caching change must never become a legality change).
"""

from __future__ import annotations

import time

import pytest
from lark.lexer import BasicLexer

from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
from slm_training.dsl.grammar.fastpath import engine as engine_mod
from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest

# Exact prefixes from PR #1171's dead_end_trace / the lexer-rebuild finding.
PREFIX8 = "root = Card([b1"
PREFIX9 = "root = Card([b1,"


def _grammar_path() -> str:
    return str((GRAMMARS_DIR / "openui.lark").resolve())


def test_load_lexer_is_cached_per_grammar_path() -> None:
    """Two lookups for the same grammar path return the identical object --
    the whole point of the fix (one BasicLexer/Scanner per grammar, not one
    per ``_lex_tokens`` call)."""
    path = _grammar_path()
    lexer_a = engine_mod._load_lexer(path)
    lexer_b = engine_mod._load_lexer(path)
    assert lexer_a is lexer_b


def test_engine_builds_scanner_at_most_once_across_many_syncs(monkeypatch) -> None:
    """Call-count regression guard: the finding measured 23,546 uncached
    ``_build_scanner`` calls for one decode-time completion-forest query.
    After the fix, the scanner is built at most once per grammar path no
    matter how many ``set_prefix``/``_lex_tokens`` calls follow."""
    engine_mod._load_lexer.cache_clear()
    engine_mod._load_parser.cache_clear()

    build_calls = {"n": 0}
    orig_build_scanner = BasicLexer._build_scanner

    def counting_build_scanner(self):
        build_calls["n"] += 1
        return orig_build_scanner(self)

    monkeypatch.setattr(BasicLexer, "_build_scanner", counting_build_scanner)

    eng = engine_mod.OpenUIIncrementalEngine()
    growing_prefixes = [
        "root",
        "root =",
        "root = Card(",
        "root = Card([b1",
        "root = Card([b1,",
    ]
    for prefix in growing_prefixes:
        eng.set_prefix(prefix)
    # A second engine instance for the same grammar must not trigger another
    # scanner build either -- the cache is keyed by grammar path, not by
    # engine instance.
    engine_mod.OpenUIIncrementalEngine().set_prefix("root = Row(")

    assert build_calls["n"] <= 1


def test_lexer_cache_speeds_up_repeated_lex_calls() -> None:
    """Wall-time benchmark: lexing the same prefix many times through the
    cached lexer must be meaningfully faster than rebuilding the scanner
    every call (the pre-fix ``self._parser.lex(prefix)`` path, still exposed
    here via the cached ``Lark`` object for a fair A/B on the same process).
    Threshold (2x) is well under the ~14.6x measured locally, to stay robust
    on a slower/loaded CI host."""
    eng = engine_mod.OpenUIIncrementalEngine()
    prefix = PREFIX9
    reps = 200

    eng._lex(prefix)  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        eng._lex(prefix)
    cached_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(reps):
        list(eng._parser.lex(prefix))  # pre-fix code path: rebuilds scanner every call
    uncached_s = time.perf_counter() - t0

    assert cached_s < uncached_s / 2, (
        f"cached lexing ({cached_s:.4f}s/{reps}) did not clear the 2x speedup "
        f"floor over the uncached Lark.lex() path ({uncached_s:.4f}s/{reps})"
    )


def test_lex_tokens_byte_identical_to_uncached_lark_lex() -> None:
    """I6: the cached path must produce the exact same token stream as the
    uncached ``Lark.lex()`` convenience method it replaces."""
    eng = engine_mod.OpenUIIncrementalEngine()
    for prefix in (PREFIX8, PREFIX9):
        cached = [(tok.type, tok.value) for tok in eng._lex(prefix)]
        uncached = [(tok.type, tok.value) for tok in eng._parser.lex(prefix)]
        assert cached == uncached


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
    """Decode-invariant proof (I3/I6): on the exact prefixes measured in the
    finding doc, ``build_completion_forest`` still returns the same
    ``coverage`` and ``n_paths`` after the lexer-caching fix as the
    pre-fix numbers recorded in
    docs/design/decode-compiler-tree-lexer-rebuild-cost-finding.json
    (``measured_direct_calls``: prefix8 n_paths=2, prefix9 n_paths=12, both
    coverage="complete"). A caching change must never become a legality
    change."""
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    prefix_ids = [tok.bos_id, *tok.encode(prefix, add_special=False)]
    forest = build_completion_forest(
        tok, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
    )
    assert forest.coverage == "complete"
    assert len(forest.paths) == expected_n_paths
