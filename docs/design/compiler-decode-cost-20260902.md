# P5: per-record compiler-tree decode cost (fixture-scale, byte-identical)

**Honesty:** fixture-scale performance work on a scratch checkpoint, one
4-CPU/no-GPU box shared with other agents. No quality, promotion, or ship
claim. Every decoded output is byte-identical before and after
(hashes below); `forwards_count` is unchanged on every record.

Machine-readable sidecar:
[`compiler-decode-cost-20260902.json`](compiler-decode-cost-20260902.json).

## Problem

The screening eval decodes each smoke record under
`STRICT_COMPILER_TREE_POLICY` (`grammar_ltr_primary`,
`compiler_decode_mode="tree"`, `output_tokenizer="lexer"`;
`src/slm_training/harnesses/model_build/eval_policy.py`). Scheduled runs
reported `compiler_ms_mean` of 23,042–34,535 ms per record
(`docs/design/continuous-openui-scheduled-*-results.json`), roughly 0.5–0.7 s
per emitted token, while
[`runtime-performance.md`](runtime-performance.md) reports ~8 ms for a warm
completion-domain query. This card attributes the gap and removes the parts
that can be removed without changing a single output byte.

## Reproduction

The committed `playground_demo/last.pt` is an output-contract v0 checkpoint
and `TwoTowerModel.from_checkpoint` refuses it (`OutputContractError`), so
the baseline uses a scratch twin of the same architecture (d_model 96,
2+3 layers, lexer output, contract v2) trained for 60 steps on the 24 smoke
records themselves (session scratch checkpoint, not committed).
Prompts: `src/slm_training/resources/data/eval/e938_role_safe_all_targets_smoke24_v1/suites/smoke/records.jsonl`.
Decode path: exactly the eval harness seam —
`apply_strict_compiler_tree_policy(model.config)` then
`generate_batch_requests([request], max_len=grammar_ltr_max_tokens)` with the
harness's `GenerationRequest` shape (runtime symbols derived from the slot
contract). `torch.set_num_threads(2)`; the denoiser is never the cost
(`denoiser_ms` = 0 on every record: every forward is an I2 exact-bypass-free
row of the compiler-tree loop, and `forwards_count` stays at 5–18).

Timing protocol: each configuration decoded records 0–7 three times in
separate processes, sequentially, with no other P5 job running; the table
reports the per-record **minimum** over the three runs. Absolute numbers are
box-specific and moved ±40 % across the session as other agents came and
went; ratios were stable.

Scratch runners (session scratch directory, not committed: `run2.py` harness-faithful
decode + cProfile; `parity_forest.py` / `parity_witness.py` differential
harnesses; `mkdoc.py` aggregation).

## Attribution (baseline cProfile, record `smoke_hero_01`, 33 tokens)

`generate_batch_requests` 13.58 s under cProfile (≈ 5.9 s unprofiled):

| cumulative | calls | function |
| ---: | ---: | --- |
| 13.41 s | 24 | `compiler_draft.build_completion_forest` (one per decode position) |
| 13.33 s | 202 | `completion_kernel.terminal_witness` |
| 13.33 s | 48,925 | `terminal_witness._eval` (bounded DFS) |
| 6.49 s | 48,935 | `CompletionSession.advance_path` → `advance` (parser fork + direct feed + semantic step) |
| 6.00 s | 1,773 | `_build_openui_completion_forest_direct` (one full forest per witness node) |

So the whole per-record cost is the **terminal-witness search** that
certifies every candidate of every completion domain: 202 root queries, each
with a 16-node allowance, expanding ~1,773 distinct kernel states, each
expansion building a full completion forest (~3.4 ms profiled, ~1.2 ms
unprofiled) and — before this card — advancing every child even after the
node allowance was already spent (48,935 `advance_path` calls for 1,773
expansions).

Suspects from the card, checked:

1. **Session binding per row** — not the cause. `pack.completion_domain`
   already binds the request-local `CompletionSession` through
   `GrammarDecodeState.bind_completion_session`;
   `completion_session_starts` = 1 per record in both `_compiler_ltr_decode_one`
   and `_compiler_ltr_decode_batch`, and `_outgoing` / `_transitions` are
   reused across positions (`completion_domain_cache_hits` 201).
2. **Forced-closure walk** — negligible here: `forced_closure` is memoized per
   `(state, room)` and follows only singleton chains
   (`completion_forced_closure_tokens` = 11 per record).
3. **`build_completion_forest` cache key includes `remaining_tokens`** — true
   but irrelevant: the domain is queried once per position, and the budget
   is part of the authority (a horizon-limited domain is a different proof,
   see `decode-invariants.md` § I2). Left untouched.
4. **`grammar_draft_window=8`** — untouched: it changes the forest paths, so
   it is not output-neutral.
5. **`_ensure_valid_openui`** — `finalize_ms` is 3–17 ms per record.

## Fixes (all output-neutral by construction)

### F1 — `terminal_witness`: stop advancing children after the verdict is fixed

`src/slm_training/dsl/grammar/fastpath/completion_kernel.py`. In a frame
whose node allowance is spent (`nodes_left <= 0`) and which has already seen
an `UNKNOWN` sibling, every remaining non-EOS child can only come back
`UNKNOWN` (nothing left to expand), `UNSUPPORTED` (room or bound prune), or a
query-local cache hit; no `SUPPORTED` entry can exist in that cache while the
query is still iterating because `SUPPORTED` unwinds every frame at once. None
of those outcomes changes `saw_unknown` or the frame's `UNKNOWN` result, which
is never persisted (`_witness_roots` stores only decided verdicts). EOS paths
are still inspected in path order because they certify without expansion.
The skipped work was a parser fork, a token feed, and a semantic step per
sibling. New session counter: `witness_children_skipped`.

Kernel counters on `smoke_hero_01` (request path): `parser_forks`
58,731 → 3,942, `transition_cache_misses` 56,957 → 2,168, `unique_states`
56,677 → 2,161, while `witness_states_expanded` (1,927) and `edges_built`
(1,773) are unchanged — the forests that decide anything are exactly the ones
still built (sidecar `record0_counters`).

### F2 — forest builder: per-build invariants computed once

`src/slm_training/dsl/grammar/fastpath/compiler_draft.py`
`_build_openui_completion_forest_direct`: the sorted terminal tuple and each
candidate's `_decision_kind` label are pure functions of the build's inputs
(tokenizer, prefix, accept set, schema, interned semantic state) and were
recomputed — including a `sorted()` over the accept set — for every admitted
path (~32 per build, 122k calls per record). They are now computed once per
build (`sorted_terminals`, `_path_decision_kind`). Labels are identical.

### F3 — `allowed_id_set`: memoized vocabulary fingerprint

`src/slm_training/dsl/grammar/fastpath/token_map.py`: the cached lookup
(`use_cache=True`, once per forest build) re-sorted and hashed the whole
`token_to_id` table to build its cache key on every call. The fingerprint is
now memoized per tokenizer (invalidated if the table object or its size
changes); uncached lookups still never touch it
(`test_uncached_native_mask_skips_cache_fingerprint`).

What was **not** done, on purpose: no candidate is added or removed, no
verification is skipped (I3), `exact_forced_token_id` and the forced-closure
walk are untouched (I2), the draft window and the domain cache key are
unchanged, and nested witness results are still query-local — reusing them
across root queries is *not* exact because the DFS verdict depends on the
node allowance and on the 64-entry query-local LRU.

## Results (records 0–7, min of 3 runs, harness-faithful request path)

| record | tokens | forwards | wall before (ms) | wall after (ms) | `compiler_ms` before | `compiler_ms` after | speedup | byte-identical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `smoke_hero_01` | 33 | 17 | 5874 | 2574 | 5784 | 2452 | 2.28× | yes |
| `smoke_button_01` | 25 | 11 | 3505 | 1506 | 3443 | 1440 | 2.33× | yes |
| `smoke_callout_01` | 29 | 15 | 5145 | 1751 | 5067 | 1676 | 2.94× | yes |
| `smoke_tabs_01` | 33 | 17 | 5550 | 1968 | 5465 | 1879 | 2.82× | yes |
| `smoke_form_01` | 33 | 16 | 5564 | 1863 | 5423 | 1769 | 2.99× | yes |
| `smoke_switch_01` | 33 | 17 | 5824 | 1891 | 5716 | 1782 | 3.08× | yes |
| `smoke_slider_01` | 29 | 14 | 5229 | 1882 | 5130 | 1777 | 2.78× | yes |
| `smoke_image_01` | 33 | 16 | 6027 | 1880 | 5949 | 1781 | 3.21× | yes |

Median per-record wall **5.56 s → 1.88 s** (2.95×; mean 5.34 s → 1.91 s) on 8 records; every record byte-identical, `forwards_count` identical, `witness_states_expanded` / `edges_built` identical. The ≤ 5 s per-record target is met on this box for all 8 records (worst record 2.57 s); before the fix 6 of 8 exceeded it. Per-run medians: before 5.87 / 5.98 / 5.56 s, after 1.98 / 2.16 / 2.08 s.

Batched path (`_compiler_ltr_decode_batch`, records 0–7 in one
`generate_batch_requests` call, single run): 26.24 s → 9.96 s (2.63×), 248 tokens,
19 forwards, identical per-row hashes. All 24 records on the per-record path:
24/24 hashes identical (before decoded as two 12-record runs, medians 6.48 s /
5.87 s; after one 24-record run, median 2.30 s, mean 2.45 s, worst 4.6 s).

### cProfile top-10 by cumulative time, record `smoke_hero_01`

Before (13.58 s profiled):

```
   ncalls  cumtime  function
        1   13.560  twotower.py:_compiler_ltr_decode_batch
       24   13.411  compiler_draft.py:build_completion_forest
       24   13.409  pack.py:_openui_completion_domain
      202   13.329  completion_kernel.py:terminal_witness
48925/201   13.328  completion_kernel.py:_eval
    48935    6.493  completion_kernel.py:advance_path
    57186    6.439  completion_kernel.py:advance
     1773    5.997  compiler_draft.py:_build_openui_completion_forest_direct
    83043    2.677  engine.py:feed_token_id
    56958    2.563  semantic_state.py:advance
```

After (6.13 s profiled):

```
   ncalls  cumtime  function
        1    6.120  twotower.py:_compiler_ltr_decode_batch
       24    6.010  compiler_draft.py:build_completion_forest
       24    6.008  pack.py:_openui_completion_domain
      202    5.926  completion_kernel.py:terminal_witness
 1967/201    5.925  completion_kernel.py:_eval
     1984    5.441  completion_kernel.py:outgoing
     1773    5.382  compiler_draft.py:_build_openui_completion_forest_direct
    24070    0.882  engine.py:set_prefix   (0.849 s of it: 1,894 full re-lex syncs)
    24050    0.765  engine.py:next_terminals
    26053    0.688  engine.py:feed_token_id
```

## Parity proof

* **Output hashes.** Every record's decoded text hashes identically before
  and after on the per-record request path (records 0–23), and the
  8-record batched call's per-row hashes match too (sidecar `extra`).
  `forwards_count`, `tokens_emitted`, `witness_states_expanded` and
  `edges_built` are identical per record.
* **Forest-level differential** (`parity_forest.py`): every
  `CompletionSession.outgoing` build during the decode of records 0–3
  (7,429 builds) was rebuilt with the pre-change builder on an independent
  control fork; `paths`, `kind` labels, `coverage` and `terminals` were
  identical in all 7,429.
* **Witness-level differential** (`parity_witness.py`): a pre-change
  `CompletionSession` ran in lockstep; every `terminal_witness` root query of
  records 0–2 (543 queries) returned the same `(status, witness)` on both
  kernels.
* Baseline outputs: session scratch `baseline_outputs.json` (24 pre-change texts; their per-row hashes are in the sidecar).

## Guard

`tests/test_models/test_compiler_decode_perf.py` decodes three smoke prompts
with a deterministic untrained fixture model (`decode_min_content=3` so the
layout has the multi-binder size of the screening records) through the same
seam and asserts the median per-record wall ≤ `PER_RECORD_BUDGET_S` (5 s).
A fixed pure-Python calibration workload is timed first; a machine more than
`MAX_CALIBRATION_SLOWDOWN` (2.5×) slower than `CALIBRATION_REFERENCE_S`
skips with the measured ratio in the reason — visible, never silent.
Honest scope: on this box the workload takes ~1.5–2.6 s per record after the
fix (test wall 5.6 s for three records, calibration 0.10–0.12 s) and
~2.0–4.4 s before it, so the 5 s budget catches gross regressions
(a return to the ~6 s pre-fix full-size cost, or worse) rather than the
fix itself; the attribution evidence is this document, not the test.

## Remaining dominant cost and next levers

After F1–F3, ≥ 90 % of a record is still `_build_openui_completion_forest_direct`
inside the witness DFS: ~1,773 distinct kernel states per record, one full
forest each. That count is fixed by the witness semantics (first-in-path-order
DFS with a 16-node allowance per candidate), so the next output-neutral gains
are per-build:

1. **Full re-lex sync on every direct-fed kernel state** (`engine.set_prefix`
   → `_full_sync`, 1,894 per record, ~16 % of the builder). A direct-fed
   engine's internal text is grammatically equivalent but not byte-identical
   to `decode_prefix(...)`, so the builder re-lexes the whole prefix. The row
   engine already skips this under the P3 `engine_ids_len` marker; extending
   that marker to kernel state records would remove it, but it inherits P3's
   equivalence rather than being exact by construction, so it is left for a
   separate change with the forest-level differential as its gate.
2. `next_terminals` / `feed_token_id` per candidate branch (~27 % of the
   builder) — Lark LALR work that is the authority itself.
3. `_generated_ast_is_complete` (official parser bridge round trip, only
   when `$END` is acceptable; 262–883 per record).

Structural change to the witness search itself (lazy path enumeration,
cross-query reuse) would change verdict order or budget accounting and is
therefore not a P5 lever.
