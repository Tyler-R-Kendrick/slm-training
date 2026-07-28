# Splitting `compiler_ms`: the Node bridge is negligible; a recursive reachability search dominates (FINDING, not a fix)

**Honesty:** `fixture_or_scratch`. Isolated, standalone-model measurement (not
the `exposure12` champion checkpoint, not the eval harness) — chosen to fit
the 3-minute wall while still exercising the real `build_completion_forest`
call path. **Not ship. Not a fix. A falsifiable measurement that corrects the
working hypothesis this iteration was framed around.**

## Why this run

Four iterations in a row measured `compiler_ms` (the timer wrapping
`build_completion_forest`) as 94.95–97.76% of the `exposure12` champion's
hero/button/callout smoke-suite decode wall time, and every doc in that chain
assumed or hypothesized that the dominant cost inside a single `compiler_ms`
call was the Node/TypeScript compiler-bridge round trip (`lang_core.py`'s
`_invoke`/`_invoke_repl`/`_invoke_once`). This session tests that hypothesis
directly by splitting a single `compiler_ms` call into its components instead
of treating it as an opaque unit.

## Method

Two bounded, CPU-only measurements against a tiny from-scratch `TwoTowerModel`
(`d_model=32`, 1 layer, seed 0) built directly from `smoke_hero_01`'s gold
OpenUI source — this avoids full eval-harness/checkpoint construction so the
whole session stays comfortably inside `MAX_RUN_SECONDS=180` per run, while
still calling the exact production function
(`build_completion_forest(..., remaining_tokens=...)`, the same call
`twotower.py`'s compiler-tree per-position loop makes).

1. **Call-count / per-call split.** Token-by-token replay of the gold source
   through `build_completion_forest`, capped at 12 calls / 90s wall so the
   run always terminates with partial data even if killed.
   `_generated_ast_is_complete` (the single call site per invocation that hits
   the Node bridge, via `lang_core.parse`) was unwrapped from its `lru_cache`
   and timed directly per call.
2. **`cProfile` of one representative slow call.** Position idx=2 of the same
   replay (~10.5s inside the 12-call batch; ~33.5s standalone in a dedicated
   profiled run — run-to-run variance on the shared sandbox noted, doesn't
   change the qualitative split) profiled end to end.

## Results

### Measurement 1 — 12-call replay, bridge vs rest

| metric | value |
| --- | --- |
| calls completed | 12 (capped) |
| `total_ms` sum / mean / median / max | 58416.22 / 4868.02 / 812.63 / 18648.35 |
| `bridge_roundtrip_ms` sum / mean / median / max | 13.98 / 1.165 / 0.844 / 3.286 |
| **bridge share of total** | **0.02%** |
| `rest_ms` sum | 58402.24 |
| **rest share of total** | **99.98%** |

Per-call totals ranged wildly (110ms → 18.6s) and did **not** correlate simply
with candidate count (call 9: 12 candidates, 18.6s; call 2: 27 candidates,
10.5s; call 10: 0 candidates, 938ms) — ruling out a simple
"more candidates ⇒ more bridge round trips ⇒ more time" story.

### Measurement 2 — `cProfile` of one 33.5s call

| function | ncalls | cumtime | % of call |
| --- | ---: | ---: | ---: |
| `pack.py:402 _tail` (recursive reachability-proof search) | 10244 (32 primitive) | 33.45s | 99.9% |
| `engine.py _sync → _refresh_accepts → lark `accepts()`` | 32270 | 16.15s | 48.2% |
| `engine.py _lex_tokens/_lex` (re-lex per node) | 47363 | 5.86s | 17.5% |
| `_generated_ast_is_complete → lang_core.parse` (**Node bridge, persistent REPL**) | 264 | **0.267s** | **0.8%** |

The Node bridge is invoked 264 times inside this single outer call (once per
`_tail` recursion node) and still totals under 1% of wall time — the
persistent-REPL default path (`lang_core.py`'s `_invoke_repl`, already the
default before this iteration) means each call pays essentially zero
subprocess-spawn cost.

## Root cause locus (not fixed this session)

`pack.py`'s `_tail_from`/`_tail` closure
(`_openui_completion_domain`, lines ~392–419) recursively proves that every
candidate completion path returned by `build_completion_forest` can actually
reach EOS within the remaining decode budget (`nodes_left=16` cap,
`@lru_cache(maxsize=64)`). Two properties make it expensive:

1. Each recursion node calls `_build_openui_completion_forest` **without**
   threading through the caller's already-synced
   `OpenUIIncrementalEngine`/state, so it builds a fresh engine and pays a
   full re-lex + re-parse from scratch every time.
2. The `lru_cache` bounding `_tail`'s own recursion is defined **inside**
   `_tail_from`'s closure, so it is empty on every one of the ~32 top-level
   ("primitive") searches within one `build_completion_forest` call — no
   sharing of proof results across sibling candidate paths.

Each fresh-engine sync calls Lark's `InteractiveParser.accepts()` to probe
reachable next-terminals, which internally copies the parser stack
non-destructively (731k `feed_token` calls, 548k `copy()` calls in this one
33.5s call) — that state-copying probe, run 32270 times, is the single
largest line item (48%).

## Falsifiable question

Is a single `compiler_ms` call dominated by Node process/bridge round-trip
overhead, or by pure-Python compiler work — and does the persistent-REPL
bridge (already the default) make the Node round trip negligible now that
the Lark lexer cache (PR #1173, iteration 1) has landed?

## Decision

**REJECT the working hypothesis carried into this iteration.** The Node
bridge is not the bottleneck — it is under 1% of cost at every granularity
measured (0.02% aggregate across 12 calls; 0.8% inside one deep-recursion
profiled call, despite 264 invocations in that single call). The dominant
cost is `pack.py`'s `_tail`/`_tail_from` recursive reachability-proof search,
specifically its per-node engine reconstruction and the resulting cascade of
Lark `accepts()` state-copy probes.

## Interpretation

This corrects, not confirms, iteration 4's next-steps note
("[c]onsider whether the Node-bridge round trip itself ... is amortizable —
a persistent bridge process instead of one-shot `subprocess.run` per call
would be a genuinely new architectural direction") — that persistent bridge
already exists and isn't the problem. It also refines iteration 1's REJECT
verdict on the Lark lexer/scanner cache: that cache targets exactly the
`_lex_tokens`/`_lex` layer measured here at ~17.5% of one call's cost — real,
but a minority contributor next to `_tail_from`'s own per-node state
reconstruction (Lark `accepts()` alone is 48%+), which is *why* caching the
lexer in isolation didn't move the wall-clock outcome: the freed budget was
spent on the same number of (or more) expensive `_tail` recursion nodes, not
fewer/cheaper ones.

## Scope note

No claim about `exposure12`'s quality or promotion status. No `decode_outcome`
or latency change — no source code was edited this session; this is pure
instrumentation of a standalone tiny scratch model, not the eval harness. No
claim that fixing `_tail_from`'s engine reuse would close the 30s wall
entirely — only that it is now the measured, evidenced hot path.

## Non-goals

No production ship claim. No code fix attempted or landed — threading engine
state through `_tail_from`'s recursion and/or hoisting its `lru_cache` above
the per-call closure boundary is a real architectural change, out of scope
for a bounded 3-minute-wall iteration per this iteration's own instructions.
No change to `decode_timeout_seconds`, deadline semantics, or outcome
classification. No GPU or real-scale training run. No claim that the Node
bridge is irrelevant everywhere — only that it is not the `compiler_ms`
bottleneck for this suite's completion-forest construction path.

## Next steps

1. If a future iteration wants to reduce `compiler_ms`, the evidenced next
   target is `pack.py`'s `_tail_from`/`_tail` closure (lines ~392–419):
   thread the caller's already-synced engine/state through each recursive
   candidate-path node instead of constructing a fresh engine per node,
   and/or hoist the `lru_cache` above the per-call closure boundary so
   sibling top-level candidate searches within one `build_completion_forest`
   call can share proof results. Both are genuine architectural changes
   (out of scope this session) needing their own bounded, falsifiable
   before/after measurement.
2. Consider whether `_tail_from`'s `nodes_left=16` budget and Lark
   `accepts()`-based reachability probe could reuse PR #1173's caching
   strategy, correctly re-scoped to the `_tail` recursion's
   fresh-engine-per-node pattern rather than the top-level call it was
   originally applied to.
3. Re-run this session's call-split protocol against the real `exposure12`
   checkpoint (not this session's tiny scratch model) if a future session
   wants production-scale confirmation that the same ~0%/~100% bridge/rest
   split holds at full model width.

## Caveats

- Measurement 2's standalone call (33.5s) is larger than the same position's
  time inside measurement 1's 12-call batch (~10.5s at idx=2) — expected
  run-to-run variance on a shared CPU sandbox; doesn't change the qualitative
  split since the bridge's absolute cost (0.267s) and share (0.8%) were
  measured directly within the same profiled run, not inferred across runs.
- This session used a tiny from-scratch model, not the `exposure12` champion
  checkpoint, to stay inside the 3-minute wall. The qualitative hot-path
  finding is a property of the `compiler_draft.py`/`pack.py`/`engine.py` code
  path itself, not this model's weights — but absolute per-call milliseconds
  are not directly comparable to iteration 4's champion-checkpoint numbers.
- `candidates_returned=27` for the profiled call is consistent with, not
  identical to, iteration 4's whole-record forward counts (17/86/31 for
  hero/button/callout) — this session profiles one isolated position, not a
  full record.

## Tests / checks

```bash
env -u NODE_OPTIONS python -m pytest -q \
  tests/test_dsl/test_grammar_capabilities.py::test_completion_domain_is_scope_local_and_witnessed \
  tests/test_harnesses/model_build/test_decode_deadline.py \
  tests/test_harnesses/model_build/test_eval_gates.py -k "timeout or stats"
# 6 passed in 1.90s
```

No source code changed this session; this subset re-confirms the
completion-domain and decode-deadline/timeout-telemetry contracts touched or
read during instrumentation are unaffected. Full `test_compiler_decode.py` /
`test_grammar_fastpath.py` suites not run (multi-minute real-model-
construction cost per repo instructions).

`python -m scripts.verify_version_stamps --check` passes — no metric/gate/
harness/matrix file was touched this session, so no component bump is
required.

## Required artifacts

This JSON/Markdown pair
(`docs/design/decode-compiler-tree-tail-reachability-search-not-bridge-20260728.{json,md}`).
