# Thrash timing Pareto recalibration (2026-08-04)

Follow-up to
[`autotrain-thrash-timing-pareto-20260803.md`](autotrain-thrash-timing-pareto-20260803.md),
which locked `screening_decode_timeout_seconds=8` from that session's
telemetry and explicitly warned: *"Never ad-hoc wall++ because a cycle
failed."* This is not that — it is the accumulated-telemetry path the same
doc prescribes for a "High (≫15%)" incomplete rate.

## What actually times out (corrected mental model)

Cycles c3/c4 of loop `continuous-openui-local`
(`docs/design/autotrain-cycle-c3-screening-decode-timeout-host-speed-20260804.md`,
`autotrain-cycle-c3-screening-decode-timeout-host-speed-20260804-results.json`,
`autotrain-cycle-c4-screening-decode-timeout-host-speed-20260804-results.json`)
were first diagnosed as "this host is ~3x slower than the 8s budget
assumed" by comparing `compiler_ms_mean` (~23s) directly against the
per-record `screening_decode_timeout_seconds` config (8s). That comparison
was wrong. `evaluate_model`'s decode wall clock is batched:
`_effective_record_decode_timeout` (`src/slm_training/harnesses/model_build/eval_runner.py:1187`)
grants `requested_seconds * chunk_record_n` for a whole chunk, and the
`screening_smoke_n=3` records here all land in one chunk of 3 — so the
*actual* effective budget observed in every scoreboard
(`effective_decode_timeout_seconds_min/max`) was `8 × 3 = 24.0s`, not `8.0s`.

## Real evidence: a razor-thin, not a 3x, miss

| Run | Arm | `total_ms_sum` (decode wall) | Effective budget | Overshoot |
| --- | --- | ---: | ---: | ---: |
| c3 | control | 24265.2ms | 24000ms | +265ms (1.1%) |
| c3 | canvas | 24000.2ms | 24000ms | cut off at wall |
| c4 | control | 24461.3ms | 24000ms | +461ms (1.9%) |
| c4 | canvas | 24000.2ms | 24000ms | cut off at wall |

All four runs on this container hit `decode_timeout_count=3/3` (100%
incomplete), but the miss was 0-461ms out of a 24000ms budget — the batch
was seconds, not minutes, from completing. This is squarely the locked
policy's own **"High (≫15%)"** incomplete-rate bucket
(`thrash_timing.incomplete_rate_high=0.15` in
`src/slm_training/resources/experiments/autotrain_climb/policy.v1.json`),
backed by four same-session, same-host samples — not the "one cycle failed"
case the 20260803 doc warns against reacting to.

## Recalibration (locked in policy, `harness.autoresearch.experiment_campaign` v178)

| Knob | Before (20260803) | After (20260804) | Rationale |
| --- | ---: | ---: | --- |
| `screening_decode_timeout_seconds` | 8 | **10** | 3×10=30s batch budget vs. observed 24.0-24.5s need: ~5.5-6s (≥20%) margin over every observed sample, still well under the `_fit_screening_decode_timeout_seconds` wall-budget ceiling (`(70 − 20 − 8) / 3 = 14.0s`, unchanged — no `MAX_RUN_MINUTES` or arm-wall change) |

`policy.v1.json` bumped `v4` → `v5`. Regression:
`tests/test_scripts/test_run_autotrain_continuous.py::test_screening_decode_timeout_has_margin_over_20260804_recalibration_evidence`
pins real margin over the worst observed sample (24.461s) so a silent
revert to 8s reproduces the same `insufficient_n` failure.
`test_fit_screening_decode_fits_arm_wall` and
`test_screening_matrix_uses_fitted_decode_and_thrash_steps` (existing,
already asserted ceilings of `<=12.0s` / `<=10.0s` — both already
anticipated a value in this range) continue to pass unchanged.

## Non-goals (unchanged from 20260803)

- Raising `MAX_RUN_MINUTES` as a thrash fix.
- Regime-epoch arm recycle.
- Counting incomplete cycles as thrash evidence for model comparison.

## Next

Retry the identical class of screening cycle (fresh manifest, same
`train_version`/`steps`) to confirm 10s clears the smoke suite on this host.
If the incomplete rate stays high after this recalibration, that is new
information for a further round — not evidence to widen further ad hoc.
