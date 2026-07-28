# Restoring `decode_stats` on a propagated compiler-tree `TimeoutError` (FIX, not ship)

**Honesty:** `fixture_or_scratch`, isolated per-record diagnostic (suite
`n=3`, 1 rep each, byte-identical recipe to the three prior docs in this
chain). **Not ship. An observability fix plus a falsifiable measurement, not
a latency fix.**

## Why this run

Iteration 3 fixed nothing new but flagged a real regression as a side
finding: once iteration 2's deadline-swallow fix (`30d8dd9`) made
`build_completion_forest` genuinely raise `TimeoutError` on the cooperative
decode deadline (instead of silently swallowing it as a fake grammar
dead-end), the propagated exception stopped carrying the `.decode_stats`
attribute `eval_runner.py`'s per-chunk timeout handler (~line 1161) expects.
Every `runtime_timeout` record's suite JSON since iteration 2 has therefore
been missing `compiler_ms` / `compiler_prefill_batches` / `trie_ms` entirely
(not zeroed — absent), which is exactly the diagnostic detail needed to
root-cause *why* the hero/button/callout smoke records can't finish inside
`decode_timeout_seconds=30`. This session fixes that gap and reruns the
isolated-record protocol to see what the restored telemetry actually shows.

## Root cause

`eval_runner.py`'s `_generate_chunk_unbounded` has two branches that populate
`decode_stats` on a timeout. The `generate_with_stats` branch (single-record
path) already attaches the stats bucket to any propagating exception at the
call site (`TwoTowerOpenUI.generate_with_stats`, `twotower.py:13998-14000`:
`except BaseException as exc: setattr(exc, 'decode_stats', stats); raise`) —
this is why the pre-existing `test_evaluate_persists_stats_when_generation_times_out`
test already passed. The `generate_batch_requests` branch — the one actually
exercised by the `exposure12` champion's compiler-tree decode path, since
`batch_size` defaults above 1 whenever `generate_batch_requests` is callable
— had no equivalent: `with collect_decode_stats() as stats: ...
predictions = generate_batch_requests(...)` let any exception, including
`TimeoutError`, propagate straight out of the `with`-block without ever
setting `exc.decode_stats`, even though `stats` (the live, incrementally-
mutated `DecodeStats` bucket) remained a valid, populated local variable at
exactly that point.

## Fix

`src/slm_training/harnesses/model_build/eval_runner.py`,
`_generate_chunk_unbounded`'s `generate_batch_requests` branch: wrap the call
in `try: ... except BaseException as exc: setattr(exc, "decode_stats",
stats); raise` inside the existing `collect_decode_stats()` block — mirrors
`TwoTowerOpenUI.generate_with_stats`'s own attach-on-raise pattern exactly.
Purely additive on the non-exception path (return value and control flow
unchanged); only changes what lands on a propagating exception.

**Invariant check (AGENTS.md § Non-negotiable architecture invariants):**
this touches only which fields land in the suite JSON after a timeout — no
change to decode timing, deadline semantics, outcome classification,
deterministic bypass, speculation/verification order, the shared
encoder↔decoder ops vocabulary, the CRDT event store, or any capability/size
lever. Confirmed safe before editing.

Regression test added:
`tests/test_harnesses/model_build/test_eval_gates.py::test_evaluate_persists_stats_when_batch_generation_times_out`
— analogous to the existing `generate_with_stats` timeout test, but exercises
the `generate_batch_requests` branch with a fake model that mutates
`get_active_stats()` then raises a bare `TimeoutError` carrying no
`.decode_stats` of its own (i.e. it only passes if `eval_runner.py` attaches
the stats, not the model).

`harness.model_build.eval` bumped `v63` → `v64`.

## Recipe

Byte-identical to all three prior docs: rebuild `lever_exposure12_v1` (107
records, 5 `abstraction_ladder`), reuse the identical cached scratch
checkpoint (sha256 verified unchanged — no retraining needed), then evaluate
each of the 3 smoke records **in isolation** (`--eval-limit 1 --eval-offset
{0,1,2}`) with the telemetry fix applied.

```bash
env -u NODE_OPTIONS python -m scripts.build_train_data --source fixture --profile strict \
  --max-records-per-parent 12 --version lever_exposure12_v1 \
  --output-root outputs/data/train

env -u NODE_OPTIONS python -m scripts.evaluate_model \
  --test-dir src/slm_training/resources/data/eval/e938_role_safe_all_targets_v2 \
  --suite smoke \
  --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
  --model twotower --device cpu \
  --checkpoint outputs/runs/exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds 30 --seed 47 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix \
  --eval-limit 1 --eval-offset <0|1|2>
```

`checkpoint_sha256=cadeb1d346d863e5a727a64847e59bfcb5c38db65fd6d5e3ef8c996c757539e9`
— matches all three prior docs bit-for-bit. `code_git_sha=7d94d66`
(iteration 3's landed doc), `code_dirty=true` (this session's fix and version
bump not yet committed at eval time, but live in the working tree for all
three runs).

## Falsifiable question

Does restoring `decode_stats` telemetry on a propagated `TimeoutError` reveal
that `compiler_ms` (`build_completion_forest`'s Node-bridge grammar-
validation round trip) or something else (`trie_ms`'s neural prefill-batch
scoring, `backbone_ms`) is the dominant remaining cost for the 3 timed-out
smoke records?

## Results: restored `decode_stats` per isolated record

| record | decode_outcome | total_ms | compiler_ms | compiler_ms % | trie_ms | backbone_ms | forwards / compiler_prefill_batches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_hero_01 (offset 0) | runtime_timeout | 30100.09 | 29425.50 | 97.76% | 157.36 | 148.36 | 17 / 17 |
| smoke_button_01 (offset 1) | runtime_timeout | 30000.11 | 28486.42 | 94.95% | 905.98 | 865.19 | 86 / 86 |
| smoke_callout_01 (offset 2) | runtime_timeout | 30000.17 | 29121.05 | 97.07% | 371.15 | 356.11 | 31 / 31 |

All three records' `metrics["decode_stats"]` is now fully populated in the
suite JSON — previously absent entirely for every `runtime_timeout` record
since iteration 2's fix landed.

## Decision

**ACCEPT the telemetry fix** — genuine, low-risk, mechanical, purely
additive on the non-exception path; regression-tested. **ANSWERED the
falsifiable question**: `compiler_ms` remains completely dominant, 94.95–
97.76% of wall time in every record, with `trie_ms` and `backbone_ms` each
under 3% in every record. No `decode_outcome` changed (all 3 records remain
`runtime_timeout`, exactly as expected — this is an observability fix, not a
latency fix).

## Interpretation

`compiler_prefill_batches_sum` tracks 1:1 with `forwards_count_sum` in every
record (17/17, 86/86, 31/31) — confirms iteration 3's read that this counter
is a correlated symptom of decode-loop iteration count, not an independent
cost driver: each per-position `build_completion_forest` call is followed by
exactly one prefill batch. The record with the most compiler-tree positions
explored (`smoke_button_01`, 86 forwards) has the *lowest* `compiler_ms`
percentage (94.95%) because its `trie_ms`/`backbone_ms` (905.98ms/865.19ms,
both scaling with `forwards_count`) eat a slightly larger share — but even
there `compiler_ms`'s raw 28486ms dwarfs everything else. There is no record
in this sample where a non-`compiler_ms` cost is remotely competitive. This
now directly measures, rather than infers from latency deltas alone, exactly
what iterations 2 and 3's interpretation sections already argued from
`build_completion_forest`'s call-site structure. Any future latency-reduction
attempt on this suite must target `build_completion_forest`'s own per-call
cost (the Node-bridge grammar-validation round trip, flagged but not yet
root-caused in `decode-compiler-tree-lexer-cache-hero-rerun-20260728.md`'s
next-steps) — bounding prefill batch width (iteration 3) or fixing lexer
caching (iteration 1) alone cannot move a 95–98%-dominated wall.

## Scope note

No claim about the `exposure12` champion's quality or promotion status. No
claim that this fix changes wall-clock latency or `decode_outcome` for any
record (by design — it only restores diagnostic fields on an already-correct
`TimeoutError` classification). No re-investigation of
`build_completion_forest`'s own internal cost breakdown yet — that's the
next open item.

## Non-goals

No production ship claim. No latency improvement claimed or found. No change
to `decode_timeout_seconds`, deadline semantics, or outcome classification.
No GPU or real-scale training run.

## Next steps

1. Root-cause `build_completion_forest`'s own per-call cost with the now-
   restored per-record telemetry as a starting instrument — e.g. does
   `compiler_ms` scale linearly with `compiler_prefill_tokens_sum`
   (4352/22016/7936 across the 3 records), or is there a large fixed
   per-call overhead independent of token count?
2. Consider whether the Node-bridge round trip itself (subprocess spawn +
   JSON marshal per `validate()` call, `src/slm_training/dsl/lang_core.py`'s
   `_invoke_once`) is amortizable — a persistent bridge process instead of
   one-shot `subprocess.run` per call would be a genuinely new architectural
   direction, not yet attempted in this doc chain.

## Tests / checks

```bash
env -u NODE_OPTIONS python -m pytest -q \
  tests/test_harnesses/model_build/test_eval_gates.py -k "timeout or stats" \
  tests/test_harnesses/model_build/test_decode_deadline.py
# 6 passed in 2.00s
```

Covers every test that exercises decode-deadline `TimeoutError` propagation
and `decode_stats` attachment, including the new
`test_evaluate_persists_stats_when_batch_generation_times_out` regression
test. Full `test_compiler_decode.py` / `test_grammar_fastpath.py` suites not
run (multi-minute real-model-construction cost per repo instructions).

`python -m scripts.verify_version_stamps --check` passes with the
`harness.model_build.eval` v63→v64 bump.

## Required artifacts

This JSON/Markdown pair
(`docs/design/decode-compiler-tree-timeout-telemetry-restore-20260728.{json,md}`).
