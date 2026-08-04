# Autotrain cycle 3 — component-plan decode-timeout replay (2026-08-04)

Per cycle 2's handoff, the driver replayed the identical frozen arm
(`continuous-loop-20260804-continuous-openui-local-8c0b60dd-c3`), reusing
cycle 2's exact checkpoints (`checkpoint_sha256` identical on both arms —
this was a true frozen replay, not a retrain).

**Result: identical failure.** All 3 smoke documents decode-timed-out on
both arms again (0/3 completed).

## What this confirms

The observed wall time landed at `latency_ms_p50_including_incomplete =
24157.64` ms against the `24.0s` chunk budget — an overshoot of ~157ms
(~0.65%). Two independent measurements of the identical checkpoints landing
in this same narrow band rules out simple transient noise as the sole
explanation; this is a reproducible, marginal-capacity signal for the
component-plan recipe family, consistent with cycle 2's diagnosis
([results](autotrain-cycle-20260804-c2-component-plan-decode-timeout.md))
and the historical 2026-08-03 run of the same family (p50 up to 23.1s).

## Why no further code change here

A third blind replay of this identical frozen arm would not produce new
information — two consecutive identical outcomes already establish the
pattern, and the loop's repeated-blocker threshold is three. Reactively
nudging the fair-share timeout formula by ~1% to clear this one recipe,
without dedicated time to test it against the thrash-calibration philosophy
(`_fit_screening_decode_timeout_seconds`'s own docstring: "not silent
wall++"), risks exactly the kind of change that codebase explicitly warns
against. The honest self-heal for *this* cycle is to **pivot the next
hypothesis** to a lighter-weight recipe that comfortably fits the current
budget (the c1-style bounds family completed at ~4.6s compiler time) rather
than a third replay of a recipe already confirmed to sit at this capacity
edge. A dedicated harness session should properly recalibrate the screening
chunk margin (e.g. per-family `decode_batch_size_max=1`, or a measured
`eval_overhead_seconds` correction) with its own tests before component-plan
is retried again.

Both checkpoints are the same local-only fixture artifacts already recorded
under cycle 2 — never promoted/synced/ship-eligible.

JSON twin: [autotrain-cycle-20260804-c3-component-plan-decode-timeout-replay.json](autotrain-cycle-20260804-c3-component-plan-decode-timeout-replay.json)
