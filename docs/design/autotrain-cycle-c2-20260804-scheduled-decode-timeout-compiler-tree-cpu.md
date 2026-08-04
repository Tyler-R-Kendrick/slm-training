# Autotrain c2 (2026-08-04 scheduled loop): dual-arm decode timeout, compiler-tree cost dominates (finding, not a fix)

**Verdict:** infrastructure failure, not scoreable, repair still required. Cycle
2 of a freshly started scheduled continuous loop
(`continuous-openui-local-20260804`, campaign
`continuous-loop-20260804-continuous-openui-local--85338931-c2`) finalized all
3/3 smoke records inside a typed decode timeout for **both** the control and
canvas arms (`decode_timeout_document_count=3`, `document_n=3`,
`completed_document_n=0`). No model evidence was produced; `primary_metric`
(`smoke.structural_similarity`) is unavailable.

## Environment

Fresh scheduled-session sandbox with **no GPU** (`nvidia-smi` is not present),
`torch==2.5.1+cpu` via `scripts/setup_dev_env.sh`, Python 3.12. `.venv/` and
`outputs/` are gitignored and not part of this commit.

## What was checked and ruled out

- `_effective_record_decode_timeout`
  (`src/slm_training/harnesses/model_build/eval_runner.py:1187`) allocates a
  fair, budget-aware per-record timeout share and is already covered by
  `tests/test_harnesses/model_build/test_eval_metric_semantics.py::test_eval_wall_fairly_caps_each_remaining_record`.
  No defect found in the allocator itself.
- `_fit_screening_decode_timeout_seconds`
  (`scripts/run_autotrain_continuous.py:261`) clamps the configured
  `screening_decode_timeout_seconds` (8.0s/record default) against the arm-wall
  budget. In this run the clamp was **not** binding (`clamp_bound=0`) — the
  configured 8.0s value is well under the theoretical ~50s/record ceiling the
  arm-wall math allows, so there was no obvious "silent wall++" available that
  would also respect the documented "if this clamp always binds, recalibrate
  the arm-share model or thrash recipe, not the wall" constraint.
- No code change was made. The driver's existing dual-arm-timeout contract
  (`tests/test_scripts/test_run_autotrain_continuous.py::test_replayed_dual_arm_timeouts_remain_inconclusive_and_require_repair`)
  — which locks in that a symmetric both-arm decode timeout must stay
  `climb_state=inconclusive` and keep demanding `repair_harness` rather than
  silently retiring to a new hypothesis — was verified intact, not weakened.

## New evidence this cycle

`scoreboard.json` shows `evaluation_policy.decode_timeout_seconds=8.0`
(per-record) and `suites.smoke.effective_decode_timeout_seconds_max/min=24.0`.
`decode_batch_size_max=3` with `decode_chunk_n=1` means all 3 smoke records
decoded as **one batched chunk**, so the effective alarm is `8.0s × 3 records
= 24.0s` total for the chunk, not per record.

The continuous-loop status matrix's decode telemetry puts `compiler_ms_mean`
at **~23.2s for both arms** (23251.2ms control, 23169.1ms canvas) —
grammar-constrained **compiler-tree compilation alone**
(`evaluation_policy.compiler_decode_mode=tree`, `strict_compiler_tree`),
before any token decode, already consumes essentially the entire 24.0s chunk
budget on this CPU-only sandbox.

This is a more specific data point than
[`autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md`](autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md),
which left open whether a dual-arm decode timeout is (a) seed-dependent
worst-case decode/parser behavior at a given candidate size, or (b) CPU
throughput vs. wall-budget headroom, without distinguishing between them. This
cycle's `compiler_ms_mean` measurement is direct support for hypothesis (b) in
this sandbox — it does not rule out (a) mattering elsewhere.

## Why this is filed as a finding, not a patch

Per [`autotrain-harness-incomplete-not-invalid-20260803.md`](autotrain-harness-incomplete-not-invalid-20260803.md):
harness/infrastructure incompletes are not model results and never
permanently invalidate an experiment — retry after the harness is fixed. This
cycle produced no model evidence to act on, and forcing the configured
decode timeout upward to make gates pass on this one CPU sandbox — without
knowing whether compiler-tree compilation is inherently this slow on CPU in
general, or specific to this fixture/model config — would risk masking real
screening-SLA regressions on faster (e.g. GPU) reference hardware. That is
architecturally significant and belongs in `improve-openui-harnesses` with
dedicated compiler-tree decode profiling, not a blind config bump from a
single scheduled cycle.

## Recommended next step

Let the driver's built-in frozen-replay handling replay the identical frozen
control/canvas arms (`retry_measurement`, already queued in
`cycle_handoff.json` — not gated on `repair_harness` per `continuous.md`'s
execution/steering-action exemption) on the next cycle of this loop, to test
whether the timeout reproduces. If it reproduces again, that is two
consecutive cycles of the same failure class in this CPU-only sandbox type —
route to `improve-openui-harnesses` for dedicated compiler-tree decode
profiling (a per-phase `compiler_ms` breakdown, or evaluating whether
`decode_batch_size_max` should default lower than 3 on CPU so one slow record
cannot exhaust its whole batch's alarm budget) instead of raising
`decode_timeout_seconds` blind.

Checkpoints (`1bc6370f...b9286e` control, `9f73b7a8...d71053b1a4` canvas) are
local, explicit no-sync, and not reusable, promotable, or ship. No champion
was confirmed or promoted from this cycle.

Machine evidence:
[`autotrain-cycle-c2-20260804-scheduled-decode-timeout-compiler-tree-cpu.json`](autotrain-cycle-c2-20260804-scheduled-decode-timeout-compiler-tree-cpu.json).
