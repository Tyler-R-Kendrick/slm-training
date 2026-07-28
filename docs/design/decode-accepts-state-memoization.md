# Memoizing Lark's `InteractiveParser.accepts()` by full LALR `state_stack`: a real, tested, measured win

**Honesty:** `fixture_or_scratch` — deterministic, in-process micro-benchmarks
and cProfile runs (constructing/driving `OpenUIIncrementalEngine`/
`_openui_completion_domain` directly on the two exact real-grammar prefixes
this stack has used throughout). No model, no checkpoint, no eval harness
involved. **Real, tested code change.**

## Task

PR #1193's named next lever: Lark's `InteractiveParser.accepts()` (12.6s /
23,530 calls, the dominant remaining witness-search cost per PR #1191's
cProfile) — investigate whether it can be soundly memoized by
parser-automaton state rather than token history, since automaton state
genuinely repeats across different token histories reaching the same LALR
state (unlike the `_tail` witness-search cache's literal-history key, which
PR #1192 measured at 0 hits). Explicitly flagged as needing its own I3/I6
safety proof before attempting.

## What I read in Lark's own source

`.venv/lib/python3.12/site-packages/lark/parsers/lalr_interactive_parser.py`
(`InteractiveParser.accepts`) and `lalr_parser_state.py` (`ParserState.feed_token`),
Lark 1.3.1.

`accepts()` copies the parser (`deepcopy_values=False`), strips callbacks
(`conf_no_callbacks.callbacks = {}`), and for each terminal `t` in
`self.choices()` (the top-of-stack state's static action-table row) probes
`feed_token(Token(t, ''))` on the copy, catching `UnexpectedToken`. The probe
`Token` is built from `LexerThread._Token`, which is just the bare `Token`
class (`_Token = Token`) — **no lexer/position state is read**, and value-stack
contents are never inspected (only their `len` for popping). So `accepts()`
is a pure function of `(parse_table, state_stack)` alone — this part of the
investigation confirms the premise.

**The key finding: top-of-stack state alone (`parser_state.position`) is
NOT a sound memoization key.** `ParserState.feed_token`'s reduce case pops
`size` frames off `state_stack` and then re-checks the *same* lookahead
token against `states[state_stack[-1]][token.type]` — the state now exposed
**below** the popped frames, which is genuinely part of the left context, not
determined by the top state alone. Two different histories that happen to
share only the top-of-stack state can expose different states below it after
a reduce and can legally disagree on whether a token is accepted.
`test_top_of_stack_alone_is_not_a_sound_key` proves this concretely with a
small synthetic Lark grammar: two branches (`"a" mid "x"` vs `"b" mid "y"`)
reduce `mid` into the *same* top-of-stack state, but `accepts()` genuinely
differs (`"X"` legal only after the `a`-branch, `"Y"` only after the
`b`-branch) — a position-only cache would silently serve one branch's answer
to the other, an I6 violation waiting to happen.

**The sound key: the full `state_stack` tuple.** `IntParseTable` (the default
LALR table, confirmed in `lalr_analysis.py`) uses plain `int` states, so
`tuple(state_stack)` is hashable. `ParserState.copy()` never copies
`parse_conf`, so every `InteractiveParser.copy()` derived from one
`Lark.parse_interactive()` call shares the identical `parse_table` object —
keying on that object (not `id()`, to sidestep id-reuse if the table is ever
evicted/GC'd) plus `tuple(state_stack)` is a provably pure, stable key.

## Fix

`src/slm_training/dsl/grammar/fastpath/engine.py`: added
`_cached_ip_accepts(ip)`, a module-level dict cache keyed on
`(parse_conf.parse_table, tuple(state_stack))` → `frozenset[str]`, LRU-bounded
at 4096 entries via a companion `OrderedDict`. `_refresh_accepts()` (used by
both `reset()` and every `_sync`) now calls it instead of `self._ip.accepts()`
directly. `accepts_cache_info()`/`_accepts_cache_clear()` are test/measurement
hooks.

## Measured: 97-98% of real `accepts()` calls served from cache; genuine wall-time win

On the exact PREFIX8/PREFIX9 prefixes PR #1171/#1172/#1191/#1192 used
(`root = Card([b1` / `root = Card([b1,`), driving
`_openui_completion_domain` (the real witness-search entry point):

| prefix | real Lark `accepts()` calls before | cache hits | cache misses (= real calls after) | hit rate |
| --- | ---: | ---: | ---: | ---: |
| PREFIX8 | 1,073 | 1,044 | 29 | 97.3% |
| PREFIX9 | 16,082 | 15,755 | 327 | 98.0% |

Wall-time (median of 5 reps, cache cleared between reps, both paths warmed
first so `_load_parser`/`_load_lexer`/regex caches are equally hot — the
`accepts()` cache is the only variable):

| prefix | cached (median, 5 reps) | uncached (median, 5 reps) | speedup |
| --- | ---: | ---: | ---: |
| PREFIX8 | 0.311s | 0.486s | 1.56x |
| PREFIX9 | 4.338s | 7.038s | 1.62x |

cProfile on PREFIX9 (`_openui_completion_domain`, single run, includes
profiler overhead so absolute numbers are inflated but the before/after
comparison is apples-to-apples): **23.28s uncached → 14.23s cached
(38.9% wall-time reduction under profiling)**. `InteractiveParser.accepts()`
drops out of the top-20-by-cumulative-time entirely post-fix; pre-fix it was
the 4th-highest-cumulative-time entry (9.08s cumulative / 16,215 calls).
Remaining cost is now genuinely lexing (`_lex_tokens`/`lex`/`next_token`,
~4.5s) and the witness search's own recursive fan-out (`_tail`/`_tail_from`),
not Lark's accept-probe.

`_sync` call counts (1,073 / 16,082) are unchanged from PR #1192's baseline —
this fix reduces *what happens inside* each `accepts()` call, not the number
of `_sync`/witness-search calls, consistent with it being a pure memoization
of an already-called operation.

## Why this is safe (I3/I6)

- `_cached_ip_accepts` is a pure cache over a provably pure function
  (`accepts()` reads only `state_stack` + the immutable per-grammar
  `parse_table`; no lexer/value-stack content). Same guarantee class as
  `_load_parser`/`_load_lexer`/`_resolve_grammar_path`'s existing
  `@lru_cache` wraps — this is a hand-rolled cache only because the key needs
  a full stack tuple plus a table-identity component that `functools.lru_cache`
  would otherwise re-derive from unhashable/mutable arguments less safely.
- `test_cached_accepts_byte_identical_to_uncached_on_real_prefixes` proves
  the cached path returns byte-identical accepted-terminal sets to the
  uncached path across 6 growing real prefixes.
- `test_top_of_stack_alone_is_not_a_sound_key` documents *why* the fix keys
  on the full stack rather than the (unsound) top-only alternative that PR
  #1193 raised as the hypothesis to investigate.
- `test_build_completion_forest_unchanged_with_accepts_cache` /
  `test_completion_domain_unchanged_with_accepts_cache` reconfirm PREFIX8/9's
  `coverage`/`n_paths`/candidate-count are unchanged (same numbers as
  `test_pack_witness_cache.py`'s golden values: PREFIX8 → 2, PREFIX9 → 12).
- The key's table-object component (not `id()`) avoids the id-reuse hazard:
  a dict key holds a strong reference, so a stale numeric id can never be
  silently reassigned to an unrelated parse table while a cache entry for it
  still exists.

## Tests added

`tests/test_dsl/test_grammar_fastpath_accepts_cache.py` (8 tests, all pass;
17.85s): byte-identical cached-vs-uncached proof, cache-hit structural proof,
the synthetic unsoundness-of-top-only-key proof, PREFIX8/PREFIX9
`build_completion_forest`/`_openui_completion_domain` invariance, and an
honest hit-rate/wall-time measurement (prints, does not hard-assert a
specific ratio — the hit rate is a property of this grammar's actual state
repetition, not guaranteed by correctness alone).

## Validation

```text
python -m scripts.repo_policy
# repo-policy: ok (tracked + untracked)

python -m scripts.verify_version_stamps --check
# version-stamps: ok (vs HEAD; 2 changed file(s), 0 component(s) touched)

python -m scripts.verify_decode_invariants
# exit 0, agent_surfaces/canonical_defaults/strict_policies/weakening_levers unchanged

ruff check src/slm_training/dsl/grammar/fastpath/engine.py \
  tests/test_dsl/test_grammar_fastpath_accepts_cache.py
# All checks passed!

pytest tests/test_dsl/test_grammar_fastpath_accepts_cache.py \
  tests/test_dsl/test_grammar_fastpath_lexer_cache.py \
  tests/test_dsl/test_pack_witness_cache.py \
  tests/test_dsl/test_grammar_fastpath_resolve_cache.py -q
# 24 passed in 36.30s
```

## Version-stamp impact

None. `src/slm_training/dsl/grammar/fastpath/engine.py` is not registered
under any component's watched `paths` in
`src/slm_training/resources/versions.json` — same precedent as PR #1173
(lexer cache) and PR #1193 (resolve cache), both of which touched this exact
file with zero `versions.json` changes.

## Scope note

- `fixture_or_scratch` per `honest-ship-eval` — process-local
  micro-benchmarks and cProfile, not a model/checkpoint/eval run.
- No `outputs/` artifacts created this session.
- This closes proposed-fix-sketch item 3 from
  [`decode-compiler-tree-witness-search-cost-finding.md`](decode-compiler-tree-witness-search-cost-finding.md)
  and the named next lever from
  [`decode-engine-init-resolve-cache.md`](decode-engine-init-resolve-cache.md).

## Named next lever

With `accepts()` no longer dominant (cProfile: it dropped out of the top-20
cumulative-time entries entirely), the remaining witness-search cost on
PREFIX9 is now split between real lexing (`_lex_tokens`/`Lark.lex`'s
`next_token`/`match`, ~4.5s of the profiled 14.2s) and the recursive
`_tail`/`_tail_from` fan-out's own Python-level overhead (4,419 recursive
`_tail` calls driving 171 `_build_openui_completion_forest` calls, each
re-deriving `_generated_ast_is_complete` → the Node DSL bridge `lang_core.parse`,
2.5s cumulative for only 171 calls). Two candidate directions for a future
bounded iteration:

1. **Lexing cost**, even with the `_load_lexer` scanner cache (PR #1173)
   live: `next_token`/`match` still run per-call per-token on every
   `_lex_tokens` invocation (23,546 calls this profiled run) because each
   call re-lexes the *whole* prefix from scratch inside `_full_sync`/
   `_incremental_sync` rather than only the incremental suffix in the cases
   where an incremental path is available — worth profiling whether
   `_incremental_sync`'s delta-only feeding is actually being taken as often
   as intended, or whether `_full_sync` fallback (full re-lex) dominates in
   this workload.
2. **The Node DSL bridge round trip inside `_generated_ast_is_complete`**
   (`lang_core._invoke`/`.parse`, 2.14s cumulative / 171 calls this run) —
   previously judged "cheap" at record-decode scale (307 calls / 0.545s in
   PR #1191's full-record cProfile) but here it's a larger fraction of a
   much shorter, isolated witness-search call; worth a fresh cProfile at
   full-record scale with the `accepts()` fix live to see whether this
   moves up the ranking now that the layer above it is smaller.

If neither of those materializes into a real, safety-proven win, the next
phase for the loop to pivot to (per this iteration's own instructions) is a
different part of the pipeline entirely — training-data build, SFT, or an
experiment matrix — rather than a fourth consecutive decode-microcost
session on this same file.

Captured: 2026-07-28T15:10:00Z
