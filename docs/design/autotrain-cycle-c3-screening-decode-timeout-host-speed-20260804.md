# Autotrain c3/c4 (continuous-openui-local, 2026-08-04 session): screening decode timeout was a razor-thin batch-budget miss, recalibrated

**Verdict:** the container is not "3x too slow" (an earlier draft of this
doc claimed that — corrected below); the miss was 0-461ms out of a 24.0s
*batch* budget. Evidence-bound recalibration applied:
`screening_decode_timeout_seconds` `8 → 10` (see
[`autotrain-thrash-timing-pareto-20260804-recalibration.md`](autotrain-thrash-timing-pareto-20260804-recalibration.md)
for the full before/after and rationale). Full per-arm numbers:
[`autotrain-cycle-c3-screening-decode-timeout-host-speed-20260804-results.json`](autotrain-cycle-c3-screening-decode-timeout-host-speed-20260804-results.json)
/
[`autotrain-cycle-c4-screening-decode-timeout-host-speed-20260804-results.json`](autotrain-cycle-c4-screening-decode-timeout-host-speed-20260804-results.json).

Cycle `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c3`
(frozen `manifest_sha256=74be089bb2215fd0db676016cd0b8aec2d5e21a9fa48fcd2a595dd25061555d9`
for `canvas`, `a20f67d2...` for `control`) and its `retry_measurement`
replay c4 (`manifest_sha256=9547f929...` canvas / `853c92e2...` control)
both reused the c2 checkpoints (frozen replay, no retraining — the AgentV
eval path now runs end to end after the cycle-2 AgentV-SDK-missing fix).
Both cycles hit `smoke:decode_timeout_count actual=3 need=0` and
`smoke:insufficient_n actual=3 need>=20` on every arm — the same shape as
`docs/design/autotrain-cycle-c3-bounds-quality-neutral.md` /
`continuous-openui-20260802-c3-results.md`.

## Corrected root cause: batched effective timeout, not 8s flat

`decode_timeout_seconds_for_role` (`src/slm_training/autoresearch/climb_policy.py:347`)
returned the (then) `screening_decode_timeout_seconds` policy default of
`8.0`. But `evaluate_model`'s decode wall clock is **per-chunk, not
per-record**: `_effective_record_decode_timeout`
(`src/slm_training/harnesses/model_build/eval_runner.py:1187`) grants
`requested_seconds * chunk_record_n`, and all `screening_smoke_n=3` records
here land in one chunk — so every scoreboard's actual
`effective_decode_timeout_seconds_min/max` was `8 × 3 = 24.0s`, not `8.0s`.
Comparing `compiler_ms_mean` (~23s) against the flat `8.0s` config (as an
earlier draft of this doc did) overstated the mismatch as "~3x too slow."
The real comparison is `decode_stats.total_ms_sum` against the `24.0s`
*effective* budget:

| Run | Arm | Decode wall (`total_ms_sum`) | Effective budget | Miss |
| --- | --- | ---: | ---: | ---: |
| c3 | control | 24265.2ms | 24000ms | +265ms (1.1%) |
| c3 | canvas | 24000.2ms | 24000ms | cut off at wall |
| c4 | control | 24461.3ms | 24000ms | +461ms (1.9%) |
| c4 | canvas | 24000.2ms | 24000ms | cut off at wall |

All four arms across both cycles missed by well under 2%, not 3x — the
batch was seconds from completing, every time.

## Why this crossed from "wontfix" to "recalibrate"

`8.0s` was deliberately locked in
[`autotrain-thrash-timing-pareto-20260803.md`](autotrain-thrash-timing-pareto-20260803.md)
with a rule: *"Never ad-hoc wall++ because a cycle failed."* c3 alone (one
cycle) would have been exactly that case, and this doc originally (wrongly)
stopped there with "no code change, host is just slow." But c4's identical
`retry_measurement` replay reproduced the same near-miss on a second,
independent run — four total arm-measurements (c3-control, c3-canvas,
c4-control, c4-canvas), 100% incomplete, all missing by <2%. That is squarely
the same locked policy's own **"High (≫15%)" incomplete-rate** recalculation
trigger (`thrash_timing.incomplete_rate_high=0.15` in `policy.v1.json`) —
accumulated same-session telemetry, not a single-cycle reaction. See
[`autotrain-thrash-timing-pareto-20260804-recalibration.md`](autotrain-thrash-timing-pareto-20260804-recalibration.md)
for the applied fix (`screening_decode_timeout_seconds` `8 → 10`,
`policy.v1.json` `v4 → v5`, `harness.autoresearch.experiment_campaign`
`v177 → v178`, regression test pinning margin over the worst observed
sample).

## Disposition

- Code/policy change applied (not a "host is just slow, do nothing" close):
  `screening_decode_timeout_seconds` `8 → 10`, still comfortably inside the
  wall-budget ceiling `_fit_screening_decode_timeout_seconds` allows
  (`(70 − 20 − 8) / 3 = 14.0s`) — no `MAX_RUN_MINUTES` or arm-wall change.
- Filed as `repair_harness` evidence per the continuous-driver handoff
  contract for both c3 and c4 so their queued `retry_measurement` actions
  could proceed.
- Next screening cycle on this loop should confirm the recalibrated 10s
  (30s batch) budget clears the smoke suite; if it still times out, that is
  new information for a further evidence-bound round, not cause to widen
  again ad hoc.

Lean is `not_applicable:screening`; climb `inconclusive`; ship `blocked` for
both c3 and c4. No checkpoint promotion or ship claim is made from either
cycle.
