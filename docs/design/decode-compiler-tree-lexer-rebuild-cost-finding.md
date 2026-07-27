# Finding: `build_completion_forest`'s per-call cost is Lark lexer rebuild, not the Node bridge

**Honesty:** `fixture_or_scratch`, isolated micro-benchmark against
`build_completion_forest` directly (no train, no eval, no checkpoint, no
model forward pass). **Not ship. Not a fix — a diagnostic finding**, same
status as
[`decode-compiler-tree-deadline-swallow-finding.md`](decode-compiler-tree-deadline-swallow-finding.md)
(PR #1171) and
[`decode-timeout-hang-seed44-steps72-finding.md`](decode-timeout-hang-seed44-steps72-finding.md).

## Task

PR #1171's own "Next steps" #2 left this open: *"Separately investigate why
each `build_completion_forest` call costs roughly 7-10s of wall time on
average ... Node-bridge round-trip cost vs. candidate-set size vs. cache
misses were not isolated this session."* This diagnostic answers that
question by calling `build_completion_forest` directly and instrumenting
every layer named in that open question.

## Reproduction

No checkpoint, train data, or `evaluate_model` run needed — `build_completion_forest`
is called directly with a real tokenizer and the exact prefix strings from
PR #1171's `dead_end_trace` (`"root = Card([b1"` / `"root = Card([b1,"`).
Environment rebuilt the same way as PR #1171 (fresh checkout, `outputs/` and
`node_modules/` gitignored):

```bash
python3.12 -m venv .venv-diag
.venv-diag/bin/pip install --quiet -e .
cd src/apps/openui_bridge && npm ci --silent   # Node DSL bridge, needed for lang_core.parse
```

**Step 1 — instrument the call chain** (`lang_core._invoke`, and
`compiler_draft._generated_ast_is_complete.cache_info()`) around direct
`build_completion_forest(tokenizer, prefix_ids, remaining_tokens=24, ...)`
calls, matching production decode (which never passes
`remaining_tokens=None`):

```python
tokenizer = DSLNativeTokenizer.build()
prefix9 = [tokenizer.bos_id, *tokenizer.encode("root = Card([b1,", add_special=False)]
build_completion_forest(tokenizer, prefix9, remaining_tokens=24, max_path_tokens=8, min_content=0)
```

**Step 2 — cProfile a warm-cache call** at the same prefix once the first
call has already populated every cache, to attribute wall time to actual
functions instead of just counting round trips.

Full scripts kept under the session scratchpad (not committed — reproduction
scaffolding, same cleanup policy as PR #1171):
`diag_build_completion_forest.py`, `diag_profile_build_completion_forest.py`.

## Measured: the Node bridge is cheap; a fully-cached repeat call is not

| call | prefix | wall_s | bridge round trips | bridge wall_s (sum) | `_generated_ast_is_complete` cache hits/misses |
| --- | --- | ---: | ---: | ---: | --- |
| cold_empty_prefix | `[bos]` | 0.681 | 10 | 0.020 | 0 / 10 |
| prefix8 | `root = Card([b1` (7 tok) | 0.956 | 23 | 0.037 | 0 / 23 |
| prefix9 | `root = Card([b1,` (8 tok) | **13.062** | 156 | 0.197 | 15 / 156 |
| prefix8_repeat | `root = Card([b1` (identical, fully cached) | **0.841** | **0** | **0.000** | **23 / 0** |

The persistent Node REPL subprocess pid stayed identical (`12340`) across
every call in-process — it was never respawned, so REPL cold-start is not in
play either.

`prefix8_repeat` is the decisive row: it re-issues the *exact same*
`build_completion_forest` call with an **already fully warm**
`_generated_ast_is_complete` cache — zero Node-bridge round trips, zero
cache misses — and still costs **0.841s**, essentially the same wall time as
the original call that made 23 real bridge round trips (0.956s, of which
only 0.037s was bridge time). The Node-bridge round-trip hypothesis from
PR #1171's open question is **ruled out**: bridge round trips are
individually ~0.4-4ms and their sum is 2-6% of total wall time even on the
worst-case (13s) call.

## Root cause: `_lex_tokens` rebuilds the Lark lexer/scanner from scratch on every incremental sync, uncached

cProfile of a second (fully warm-cache) call to `build_completion_forest` at
`prefix9` (`remaining_tokens=24`) — 95.6M function calls, ~41s under
profiler overhead (the profiler itself adds ~3x; the matching unprofiled
call above measured 13.06s) — attributes the cost to:

```
build_completion_forest (compiler_draft.py:2246)
  -> completion_domain (grammar_capabilities.py:300)
    -> _openui_completion_domain (pack.py:273)
      -> _tail_from (pack.py:392), _tail (pack.py:402)     4,419 calls (recursive, bounded nodes_left=16)
        -> _build_openui_completion_forest (compiler_draft.py:1480)   171 calls
          -> OpenUIIncrementalEngine._sync (engine.py:177)             16,082 calls
            -> _lex_tokens (engine.py:99): self._parser.lex(prefix)    23,546 calls, 27.1s cumulative
              -> lark/lexer.py:536 lex -> :614 next_token -> :605 scanner -> :592 _build_scanner
                -> lark/lexer.py:334 _create_unless                    23,546 calls, 14.7s cumulative
                  (regex compilation: 6.4M calls to re._compile, 5.5M calls to re.match)
```

`_lex_tokens` (`engine.py:99-110`) calls `self._parser.lex(prefix)` — Lark's
**standalone** `.lex()` convenience API — passing the **entire current
prefix** on every `_full_sync` (`engine.py:124-147`) and every
`_incremental_sync` (`engine.py:149-175`) call. `self._parser` is the
shared `Lark` grammar object from `_load_parser` (`engine.py:43-53`,
`@lru_cache(maxsize=4)`), so the *grammar* is cached — but `Lark.lex()`
itself is documented as a one-shot convenience method: each call goes
through `lark/lexer.py`'s internal (uncached) `_build_lexer` ->
`_build_scanner` -> `_create_unless` path, which recompiles the terminal
disambiguation regex tables from scratch every time. Nothing in
`OpenUIIncrementalEngine` caches or reuses a built `Lexer`/`Scanner`
instance across `_sync` calls, even though the grammar and its terminal set
never change within (or across) a decode session.

That per-call lexer-rebuild cost (0.62ms average, from 14.7s / 23,546 calls)
is individually small but is invoked at every incremental parse step inside
the *recursive* candidate-witness search: `pack.py`'s `_tail_from`/`_tail`
(the same bounded 16-node lookahead-then-verify search described in I3,
[`docs/design/decode-invariants.md`](decode-invariants.md)) calls
`_build_openui_completion_forest` up to 171 times for a **single** top-level
`build_completion_forest` invocation at this dead-end prefix, and each of
those calls drives many `_sync` calls of its own. The multiplication —
23,546 uncached lexer rebuilds for one decode-time completion-forest query —
is what turns a sub-millisecond-scale operation into 7-13 seconds of wall
time, independent of the Node bridge, independent of the
`_generated_ast_is_complete` cache, and largely independent of
`compiler_candidates_sum` (which only counts admitted paths, not the much
larger internal `_tail` exploration count).

## Interpretation vs. PR #1171

PR #1171's finding correctly identified that `compiler_ms` (not
`denoiser_ms`, not `forwards_count`) is the wall-clock sink, and that the
per-record cost scales with `compiler_prefill_batches`. That still holds.
What this diagnostic adds/corrects: the internal breakdown of
`compiler_ms` speculated in that doc's "Next steps" ("Node-bridge round-trip
cost vs. candidate-set size vs. cache misses") points at the wrong
subsystem. The Node bridge (`lang_core.parse`, the actual out-of-process
Node subprocess) is fast and well-cached; the uncached Lark lexer
reconstruction inside the *in-process* `OpenUIIncrementalEngine._sync` path
is the dominant cost, and it compounds with the recursive
`_tail_from`/`_tail` witness search's call count rather than with
candidate-set size directly (`prefix9`'s 6.8x higher `_generated_ast_is_complete`
call count over `prefix8` corresponds to a 13.7x wall-time increase — worse
than linear, consistent with recursive `_tail` fan-out multiplying an
uncached per-node cost rather than a single flat per-candidate cost).

## Why this is a finding, not a patch

Same precedent as the two prior findings this follows up on: the affected
file (`engine.py`) backs `dsl.grammar_capabilities` (watched in
`src/slm_training/resources/versions.json` via `pack.py` /
`compiler_draft.py`) and is exercised by every default constrained-decode
path (I3's speculative/lookahead witness search). A real fix (caching a
built `Lexer`/`Scanner` per grammar the way `_load_parser` already caches
the `Lark` object, or replacing the `_lex_tokens`-per-`_sync` full-prefix
re-lex with genuinely incremental lexing) needs a benchmark-backed unit test
and should go through `improve-openui-harnesses`, not be patched blind
here.

## Proposed fix sketch (not applied this session)

1. Cache a built `Lexer` (or the object `Lark.lex()` constructs internally)
   per grammar path alongside `_load_parser`'s cached `Lark` object
   (`engine.py:43-53`), and have `_lex_tokens` (`engine.py:99`) reuse it
   instead of calling the standalone `self._parser.lex(prefix)` convenience
   API, which rebuilds the scanner every call.
2. Alternatively/additionally, make `_incremental_sync` (`engine.py:149`)
   truly incremental at the lex layer — it already tracks `_fed_token_count`
   and only feeds new tokens to the `InteractiveParser`, but it still
   re-lexes the *entire* prefix from scratch (`_lex_tokens(prefix)` on the
   full string, `engine.py:152`) before slicing to the new tokens.
3. Re-run this diagnostic's `prefix9` case after the fix and confirm
   `_lex_tokens`/`_build_scanner` call counts and `compiler_ms` both drop
   without changing `n_paths`/`coverage` (grammar/decode semantics must stay
   byte-identical — this is a caching change, never a legality change per
   I6/I3).
4. Re-run the PR #1171 isolated per-record eval protocol (or the seeded
   multi-rep `lever-hard-decode-timeout-wall` protocol) to see whether
   `compiler_ms` drops enough for the exposure12 quality-champion hero to
   finish inside `decode_timeout_seconds=30` without needing the deadline
   fix from PR #1171 at all.

## Scope note

This is a single-process micro-benchmark (n=1 rep per prefix, 4 prefixes),
not a multi-rep confirm, and it does not by itself prove this is the *only*
contributor to `compiler_ms` on every record — but the ~0% marginal cost of
a fully-cached repeat call (`prefix8_repeat`) is sufficient on its own to
rule out the Node bridge and the `_generated_ast_is_complete` cache as
primary cost drivers, and the cProfile breakdown directly attributes the
remaining cost to `_lex_tokens`'s uncached `Lark.lex()` calls.

## Cleanup note

`.venv-diag` and `src/apps/openui_bridge/node_modules/` created for this
diagnostic are not committed (both gitignored); the two diagnostic scripts
live under the session scratchpad, not the repo, since they are
reproduction scaffolding rather than a reusable harness tool.

Captured: 2026-07-27T20:35:52.529695+00:00
