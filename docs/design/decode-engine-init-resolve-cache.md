# Caching `OpenUIIncrementalEngine.__init__`'s resolved grammar path: a real, tested, measured 400x reduction in `Path.resolve()` calls

**Honesty:** `fixture_or_scratch` — deterministic, in-process micro-benchmark
(constructing `OpenUIIncrementalEngine()` in a loop with `Path.resolve`
monkeypatched to a counting wrapper). No model, no checkpoint, no eval
harness involved. **Real, tested code change** — a pure path-string
memoization, exactly as scoped by the named next lever.

## Task

PR #1192's named next lever
([`decode-compiler-tree-witness-search-cache-hoist.md`](decode-compiler-tree-witness-search-cache-hoist.md)):
`OpenUIIncrementalEngine.__init__`'s per-construction `Path.resolve()` /
`os.path.realpath()` cost (10,967 calls / ~5.1s combined in PR #1191's
cProfile) — cache the resolved grammar path once (module-level, keyed on the
raw `grammar_path` string).

## Call site confirmed

`src/slm_training/dsl/grammar/fastpath/engine.py`, `OpenUIIncrementalEngine.__init__`:

```python
self._parser = _load_parser(str(path.resolve()))
self._lexer = _load_lexer(str(path.resolve()))
```

`_load_parser`/`_load_lexer` are already `@lru_cache(maxsize=4)`-wrapped, but
their cache key is the *resolved* path string — so `path.resolve()` (which
calls `os.path.realpath()` internally) runs **twice per construction**,
unconditionally, before either cache's key is even computed. The existing
caches were never broken; the resolve call simply sat upstream of them,
untouched by PR #1173's lexer-cache fix or PR #1192's witness-cache hoist.

## Fix

Added a module-level cache keyed on the raw (pre-resolve) path string:

```python
@lru_cache(maxsize=8)
def _resolve_grammar_path(grammar_path: str) -> str:
    return str(Path(grammar_path).resolve())
```

`__init__` now calls it once and reuses the single resolved string for both
`_load_parser` and `_load_lexer` lookups, instead of resolving twice
directly.

## Measured: 400 → 1 `Path.resolve()` calls for 200 constructions

| scenario | `Path.resolve()` calls (200 constructions, same grammar path) |
| --- | ---: |
| before (`git stash`) | 400 (2 per construction) |
| after | 1 |

Reproduced directly: monkeypatched `pathlib.Path.resolve` to a counting
wrapper, cleared all three engine-module caches (`_load_parser`,
`_load_lexer`, `_resolve_grammar_path`), then constructed
`OpenUIIncrementalEngine()` 200 times for the default `openui.lark` grammar
path. Pre-fix and post-fix numbers captured in the same process family via
`git stash`/`git stash pop` on the `engine.py` diff.

This directly confirms the predecessor finding's cProfile line (`Path.resolve()`/
`os.path.realpath` firing on every construction, pre-lru_cache-key) and shows
the fix collapses it to the minimum possible: one real resolve per distinct
grammar path used in the process, cached for the process's remaining
lifetime.

## Why this is safe (I3/I6)

Pure string→string memoization of a deterministic, side-effect-free
operation. Grammar files are never swapped or symlink-retargeted mid-process,
so caching the resolution for a fixed raw path string can never become
stale. The cache key is the *raw* `grammar_path` string as passed to
`__init__`, not the resolved value — two different raw strings that happen
to resolve to the same real path are each looked up independently and
correctly (no incorrect collapsing of distinct entries), proven by
`test_resolve_cache_distinguishes_different_grammar_paths` using
`openui.lark` and `toy_layout.lark`. No witness-search, parsing, or
decode-legality code was touched; `test_engine_still_functions_after_resolve_caching`
confirms `set_prefix`/`next_terminals` still work end to end through the
cached-resolve path.

## Tests added

`tests/test_dsl/test_grammar_fastpath_resolve_cache.py` (4 tests, all pass;
0.32s):

1. `test_resolve_call_count_drops_after_caching` — the call-count regression
   guard (400 → ≤2 for 200 constructions). **Sanity-checked to fail
   meaningfully pre-fix**: `git stash`ing the `engine.py` change and rerunning
   this test reproduces the pre-fix 400-call count, well over the post-fix
   threshold.
2. `test_resolved_path_value_matches_uncached_resolve` — correctness: the
   cached value equals `Path(grammar_path).resolve()` computed directly.
3. `test_resolve_cache_distinguishes_different_grammar_paths` — cache
   correctness across two different `grammar_path` strings (`openui.lark` vs
   `toy_layout.lark`): distinct resolved values, distinct cache entries,
   and the downstream `_load_parser`/`_load_lexer` objects for the two
   engines are correctly distinct too (proves the resolve cache feeds the
   right key downstream, not a collapsed one-size-fits-all string).
4. `test_engine_still_functions_after_resolve_caching` — end-to-end smoke
   that the engine still parses/accepts normally through the cached path.

## Validation

```text
python -m scripts.repo_policy
# repo-policy: ok (tracked + untracked)

python -m scripts.verify_version_stamps --check
# version-stamps: ok (vs HEAD; 2 changed file(s), 0 component(s) touched)

python -m scripts.verify_decode_invariants
# exit 0, agent_surfaces/canonical_defaults/strict_policies/weakening_levers unchanged

ruff check src/slm_training/dsl/grammar/fastpath/engine.py \
  tests/test_dsl/test_grammar_fastpath_resolve_cache.py
# All checks passed!

pytest tests/test_dsl/test_grammar_fastpath_resolve_cache.py -q
# 4 passed in 0.32s

pytest tests/test_dsl/test_grammar_fastpath_lexer_cache.py \
  tests/test_dsl/test_pack_witness_cache.py \
  tests/test_dsl/test_grammar_fastpath_resolve_cache.py -q
# 16 passed in 32.09s
```

## Version-stamp impact

None. `src/slm_training/dsl/grammar/fastpath/engine.py` is not registered
under any component's watched `paths` in
`src/slm_training/resources/versions.json` (confirmed by grep across the
file). This mirrors PR #1173's own precedent — that PR also modified this
exact file (the lexer-cache fix) and made zero `versions.json` changes,
confirmed by inspecting commit `20b658f`'s diff (no `versions.json` hunk).
`verify_version_stamps --check` reports `0 component(s) touched`, consistent
with that precedent.

## Scope note

- Diagnostic + fix, `fixture_or_scratch` per `honest-ship-eval` — a
  process-local `Path.resolve()`-call-count micro-benchmark, not a
  model/checkpoint/eval run.
- No `outputs/` artifacts created this session.
- This closes proposed-fix-sketch item 2 from
  [`decode-compiler-tree-witness-search-cost-finding.md`](decode-compiler-tree-witness-search-cost-finding.md)
  and the corresponding named next lever from
  [`decode-compiler-tree-witness-search-cache-hoist.md`](decode-compiler-tree-witness-search-cache-hoist.md).

## Next lever

Lark's `InteractiveParser.accepts()` (12.6s / 23,530 calls in PR #1191's
cProfile) remains the dominant unaddressed cost — item 1 from both
predecessor docs' "next lever" sections. A real win needs memoizing
`accepts()` by parser-automaton state (not token history): automaton state
genuinely repeats across different token histories that reach the same LALR
state, unlike the `_tail` witness-search cache's literal, strictly-monotonic
history key (which PR #1192 measured at 0 hits by construction). This is a
bigger, Lark-internals-facing change and needs its own safety proof (I3/I6:
witness content and `coverage` must stay byte-identical before/after) —
deferred to a future bounded iteration via `improve-openui-harnesses`, not
attempted here to avoid crowding out this session's primary, already-shipped
lever.

Captured: 2026-07-28T13:45:00Z
