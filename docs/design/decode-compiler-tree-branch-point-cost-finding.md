# Finding: compiler-tree decode cost is branch-point-driven, not a bug

**Honesty:** `fixture_or_scratch`. Diagnostic reproduction only. `not_ship:
true`, `is_fix: false` (no production code changed — this session found no
algorithmic defect to fix). One regression test was added to pin the
invariant this session's profiling established.

**Routed from:** the `improve-openui-harnesses` follow-up requested by
[`continuous-openui-local-gd6j83-c2-dual-arm-decode-timeout.md`](continuous-openui-local-gd6j83-c2-dual-arm-decode-timeout.md)
and its
[`continuous-openui-local-gd6j83-c3-retry-still-times-out.md`](continuous-openui-local-gd6j83-c3-retry-still-times-out.md)
retry (both left untouched by this change, per instruction): after the
[metering-gap fix](compiler-tree-forced-closure-decode-metering-gap.md)
confirmed `compiler_ms` was already an honest measurement for
`compiler_decode_mode="tree"` on both cycle-2/3 arms, the open question was
*why* a single seed-100002 `wf_smoke_v2` smoke record legitimately costs
~23s of compiler-tree decode — an algorithmic bug (missing memoization,
quadratic-in-length blowup, a pathological grammar shape), or genuine bounded
cost from this decode mode's own design.

## Constraint on this session's reproduction

The campaign checkpoints referenced by the gd6j83-c2/c3 docs
(`becbf08d…2adf7e`, `6953396f…f36597`) are **not present in this workspace** —
there is no `outputs/` directory at all (no `outputs/autoresearch/`,
`outputs/runs/`). This session cannot replay the identical frozen arm. Instead
it exercises the exact same production code paths
(`TwoTowerModel._compiler_ltr_decode_one`, `build_completion_forest`,
`CompletionSession`/`outgoing()`/`terminal_witness`) directly against real
`wf_smoke_v2` fixture records
(`src/slm_training/resources/data/train/wf_smoke_v2/records.jsonl`, 101
records) using a small untrained probe model (`d_model=32`, 1 layer, CPU) —
sufficient because the completion-forest/grammar-authority cost is a function
of the **legal candidate set at each grammar state** (the pack's schema plus
the record's own slot contract), not of the model's learned weights.

## What was measured

### 1. Node/backtrack bounds are intact (re-confirms the metering-gap doc, direct evidence this time)

`CompletionSession.terminal_witness`'s `node_budget` (16) and
`LatticeSearchState.backtrack_limit` (8) are enforced exactly as read in the
metering-gap doc. This session adds **direct instrumented confirmation**
rather than source inspection alone (see §3).

### 2. Natural (model-driven) decode cost on real fixture records

`_compiler_ltr_decode_one(..., mode="tree")` run to completion (EOS reached
via the untrained probe model's own candidate choices, not gold-forced) on a
sample of `wf_smoke_v2` records spanning the fixture's full length range
(61–286 chars, 21–101 gold tokens):

| record | gold_len_tokens | tokens_emitted | wall_ms | compiler_ms |
| --- | --- | --- | --- | --- |
| `train_callout_01` | 21 | 9 | 1860.2 | 1294.5 |
| `train_callout_01_syn_0` | 21 | 11 | 1267.0 | 1252.7 |
| `train_callout_01_syn_1` | 21 | 11 | 1255.4 | 1238.6 |
| `train_form_three_controls_01_syn_2` | 94 | 10 | 1185.2 | 1178.0 |
| `train_form_three_controls_01` | 94 | 13 | 1347.4 | 1326.2 |
| `train_metrics_01_aug_dir` | 101 | 15 | 1446.9 | 1423.7 |

Two things stand out: **(a)** `compiler_ms` per record does not track
`gold_len_tokens` (21-token and 101-token gold records cost about the same);
it tracks `tokens_emitted` from the *decode's own trajectory* (9–15 tokens
here), and **(b)** per-token cost is roughly 90–150ms — much higher than the
flat ~0.6ms/step measured in §3 below along a low-ambiguity path. This is the
first piece of direct evidence that cost is concentrated at genuinely
ambiguous grammar branch points, not spread uniformly per token.

### 3. Depth-scaling probe: no missing-memoization / quadratic-in-length signature

To directly test the "redundant recomputation across steps" hypothesis, this
session replayed a record's own gold token sequence as `_initial_prefix` at
increasing depth `k`, then measured the cost of exactly **one more** real
`_compiler_ltr_decode_one` step from that depth (same session/state-interning
machinery a real multi-step decode would reuse):

```
record=train_metrics_01_aug_dir gold_len_tokens=101
  k=  10 prefix_len=  11 wall_ms=   1.9 compiler_ms=  0.8
  k=  20 prefix_len=  21 wall_ms=   1.8 compiler_ms=  0.6
  k=  40 prefix_len=  41 wall_ms=   2.8 compiler_ms=  0.9
  k=  60 prefix_len=  61 wall_ms=   3.7 compiler_ms=  1.0
  k=  80 prefix_len=  81 wall_ms=   4.9 compiler_ms=  0.6

record=train_form_three_controls_01 gold_len_tokens=94
  k=  10 prefix_len=  11 wall_ms=   1.5 compiler_ms=  0.6
  k=  20 prefix_len=  21 wall_ms=   1.8 compiler_ms=  0.6
  k=  40 prefix_len=  41 wall_ms=   2.6 compiler_ms=  0.7
  k=  60 prefix_len=  61 wall_ms=   3.3 compiler_ms=  0.6
  k=  80 prefix_len=  81 wall_ms=   4.2 compiler_ms=  0.6
```

(`k=0` excluded from the table: it pays a one-off ~500ms `make_grammar_state`
/ Lark-parser construction cost unrelated to decode search, not decode-step
cost — see the `_new_grammar_states`/`make_grammar_state` frame in the
cProfile trace below.)

`compiler_ms` for one more step stays flat (~0.6–1.0ms) across an 8x growth
in prefix depth (10 → 80 tokens), along a path that mostly hits
forced-closure singleton chains (the I2 deterministic bypass) rather than
live branch points. This **rules out** a position-dependent /
missing-memoization defect: if `_materialize_prefix`'s ancestor walk, the
`CompletionSession`'s state-interning, or the domain/witness caches had
regressed to redo work proportional to how deep the prefix already is, this
probe would show growing — not flat — per-step cost.

### 4. Where the natural-decode cost actually goes (cProfile)

Profiling the natural (model-driven) decode of `train_callout_01` (9 tokens
emitted, `compiler_ms≈1.2–1.3s` across the sample runs) shows the cost
concentrated in the packed completion kernel's witness search, not in a
single runaway operation:

```
ncalls  cumtime  filename:lineno(function)
     6    2.645   dsl/pack.py:287(_openui_completion_domain)
    40    2.581   dsl/grammar/fastpath/completion_kernel.py:478(terminal_witness)
9513/40   2.581   dsl/grammar/fastpath/completion_kernel.py:499(_eval)          # recursive witness DP
 11310    1.344   dsl/grammar/fastpath/completion_kernel.py:344(advance)
   306    1.174   dsl/grammar/fastpath/completion_kernel.py:430(outgoing)
   283    1.165   dsl/grammar/fastpath/compiler_draft.py:1646(_build_openui_completion_forest_direct)
 11289    0.704   dsl/grammar/fastpath/semantic_state.py:574(advance)
 17199    0.485   dsl/grammar/fastpath/engine.py:673(feed_token_id)
```

For just **6** outer `build_completion_forest` calls (6 real decode-loop
iterations), the witness search makes **40** top-level `terminal_witness`
queries, **9,513** recursive `_eval` expansions, **306** `outgoing()` calls
(283 of them real forest-build cache misses), and **17,199** low-level
`feed_token_id` calls. This matches `node_budget=16` bounding *each*
top-level candidate's exploration (283 forest builds / 40 witness roots ≈ 7
per root — consistent with a handful of live top-level candidates per branch
point, each budgeted up to 16 nodes) — i.e. genuinely bounded per branch
point, but multiplied by however many branch points a given decode
trajectory visits. No single call dominates; the cost is the sum of many
cheap-but-real Lark-parse-plus-semantic-state operations
(`semantic_state.py:574 advance`, `engine.py:673 feed_token_id`,
`lalr_parser_state.py:67 feed_token`), exactly the "genuinely non-trivial,
repeated ... invoked once per token (or per candidate)" cost the
metering-gap doc already characterized in the aggregate — this session
additionally shows *why* it varies so much record-to-record and
decode-to-decode: it is proportional to branch-point count, not token count.

## Verdict

**Legitimate cost, not an algorithmic bug** (task item 4, not item 3):

- The search kernel's own bounds (`node_budget`, `backtrack_limit`, deadline
  checks, structural state interning) are intact and were re-verified with
  direct instrumented evidence, not just source review.
- A dedicated depth-scan found **no** position-dependent cost growth — the
  missing-memoization / quadratic-in-document-length hypothesis is ruled
  out by direct measurement.
- The real cost driver is **branch-point count**: how many grammar states
  along the actual decode trajectory have more than one live candidate path
  requiring a bounded (`node_budget≤16`-per-candidate) terminal-witness
  proof. This is a property of the *decode trajectory* (which depends on
  what the specific trained checkpoint's ranking picks at each ambiguous
  junction), not simply a static property of the record's gold text length.
  A model whose choices wander into more grammar states with many live
  candidates — whether because of undertrained/miscalibrated ranking at a
  particular seed, or a genuinely branch-rich record structure — pays this
  bounded-per-point cost more times, and the total is not capped by
  anything except the decode length (`max_target_len`) and the cooperative
  wall-clock deadline (`check_decode_deadline`), which **is** honored
  throughout (a slow decode fails closed with `TimeoutError`, never hangs).
- This reconciles with, and does not contradict, the metering-gap doc's own
  real single-record numbers on actual campaign checkpoints (14.8s–38s per
  record) — tens of seconds is the expected order of magnitude once a
  decode visits more than a handful of genuinely ambiguous branch points at
  ~50-500ms each, not evidence of a runaway/exponential search.

## What this session does NOT claim

Because the actual gd6j83-c2/c3 campaign checkpoints are unavailable here,
this session cannot prove those *specific* seed-100002 records' decodes are
dominated by branch-point count as opposed to some other seed-specific
effect. It establishes the general mechanism (bounded-per-point,
trajectory-dependent, non-quadratic cost) on real fixture content using the
real production code paths, which is as far as reproduction can go without
the frozen checkpoints.

## Policy recommendation (no blind timeout bump)

Per the explicit instruction not to raise `decode_timeout_seconds` without
evidence tied to a measured property, this session does **not** change
`decode_timeout_seconds`, `_effective_record_decode_timeout`, or any ship
gate. Two narrower, already-evidence-backed observations for whoever next
replays this frozen manifest with the real checkpoints:

1. `completion_unique_states`, `completion_edges_built`,
   `completion_witness_states_expanded`, and `completion_full_sync_fallbacks`
   are **already** captured per-record in `DecodeStats` and already
   aggregated by `aggregate_stats()` (`src/slm_training/models/decode_stats.py`),
   and `_persist_decode_progress`
   (`src/slm_training/harnesses/model_build/eval_runner.py`) already writes
   them to `decode_progress.json` on exactly this kind of decode-timeout
   interruption. A future replay of this frozen arm should read that file's
   `completion_*` counters directly — no new profiling harness needed — to
   confirm or refute that these specific records' cost is branch-point-count
   driven (this finding's story) versus something else. This session found no
   gap here; nothing needed adding.
2. `_effective_record_decode_timeout`
   (`src/slm_training/harnesses/model_build/eval_runner.py:1209`) computes a
   fair wall-clock share **independent of `compiler_decode_mode`** — it does
   not already reserve more budget for `tree` mode's measured
   costlier-per-branch-point profile than for cheaper modes. If a future
   session, replaying the real frozen checkpoint, confirms these records
   legitimately need more wall time than the allocator currently grants
   (rather than an anomaly), a mode-aware fair-share reservation tied to the
   measured branch-point cost (not a flat guess) would be the smallest
   correct lever — proposed here, not implemented, since it needs the actual
   checkpoint's branch-point counts to size correctly.

## Regression test added

[`tests/test_models/test_compiler_decode.py::test_compiler_tree_per_step_cost_does_not_grow_with_prefix_depth`](../../tests/test_models/test_compiler_decode.py):
replays an increasingly deep gold prefix of a branchy fixture record
(`Card` with four `TextContent` children) and asserts one more real
`_compiler_ltr_decode_one("tree")` step costs roughly the same `compiler_ms`
regardless of prefix depth (generous ratio bound, passes today with a wide
margin; would fail hard under a genuine O(n)-per-step regression). Verified
stable across repeated runs (3/3 passes, ~2.1–3.2s each).

## Test results

- `tests/test_models/test_compiler_decode.py` — 235 passed.
- `tests/test_dsl/test_exact_forced_horizon.py tests/test_models/test_compiler_decode.py` — 237 passed (confirms the just-merged metering-gap regression test still passes).
- `tests/test_dsl/test_completion_kernel.py` — 43 passed.
- `tests/test_scripts/test_run_autotrain_continuous.py` — 222 passed, 1 skipped;
  `test_replayed_dual_arm_timeouts_remain_inconclusive_and_require_repair`
  passes unmodified — the dual-arm-timeout contract this investigation must
  not weaken is intact.
- `python -m scripts.repo_policy` — ok.
- `python -m scripts.verify_version_stamps --check` — ok, 0 components touched
  (the new test file is not a watched path for any component; no bump
  required).
- `python -m scripts.refresh_test_cases --check --changed` — no mirrored
  resource cases touched.
- `.githooks/check-changed` — ruff + py_compile + targeted pytest all pass,
  zero new failures.

## Conclusion

No algorithmic bug found in the compiler-tree completion-forest/witness
search: its bounds are correctly enforced, and its cost does not grow with
document position. The recurring ~23s/record cost this failure class keeps
surfacing is genuine, bounded-per-branch-point, trajectory-dependent decode
cost — not a metering artifact (already fixed upstream) and not a runaway
search (ruled out here). Closing this as a legitimate-cost finding per task
item 4; the open next step (item 2 in the policy recommendation above) is
gated on having the actual frozen checkpoints available to size a mode-aware
timeout reservation correctly, which this workspace does not have.

Machine evidence:
[`decode-compiler-tree-branch-point-cost-finding.json`](decode-compiler-tree-branch-point-cost-finding.json).
