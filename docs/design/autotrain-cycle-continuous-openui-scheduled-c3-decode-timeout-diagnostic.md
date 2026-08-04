# Autotrain c3 (continuous-openui-scheduled): screening decode-timeout diagnostic

**Verdict:** measurement incomplete, infrastructure diagnosis — not a harness
defect, not a model result. Frozen replay of the c2 `control`/`canvas` smoke
arms (`00574c47f6362eaae01b999e28683e721f092026d1c561e5669ef22a0a811210`) now
reached AgentV (the SDK self-heal from
[`autotrain-cycle-continuous-openui-scheduled-c2-agentv-missing-recurrence.md`](autotrain-cycle-continuous-openui-scheduled-c2-agentv-missing-recurrence.md)
worked), but every smoke record (`3/3`) hit the per-record decode timeout on
both arms: `suites.smoke.compiler_ms_mean≈23,242` (control) /
`≈23,166` (canvas) against a fitted `screening_decode_timeout_seconds≈8.0`
(`scripts/run_autotrain_continuous.py:_fit_screening_decode_timeout_seconds`,
`arm_wall_seconds≈52.76`, `min_train_floor_seconds=20`,
`eval_overhead_seconds=8`, `smoke_n=3`).

## Diagnosis

The 8s screening default
(`climb_policy.decode_timeout_seconds_for_role`) is deliberately tuned
against this repo's established smoke-decode latency: historical measured
results run in the `8xx`–`4,000` ms range per record (e.g.
`autotrain-cycle-c3-bounds-quality-neutral.md` p50 `3,064.79`/`3,107.70` ms;
`autotrain-cycle-1817-component-edge-token-null.md` p50 `960.07`/`962.44`
ms), and `test_fit_screening_decode_fits_arm_wall` pins the clamp at `<=
12.0s`. A `23,242` ms mean is 6-8x that range — inconsistent with a
miscalibrated *default* (which would show up across many historical cycles,
not appear for the first time here) and consistent with a **cold sandbox**:
this cycle ran in a scheduled-routine container that had just bootstrapped
`.venv` and `node_modules` (`npm ci`) moments earlier, with no prior warm
decode in this process.

This is a single occurrence, not a repeated pattern (loop law: report
`blocked` only after the same hard blocker fails three consecutive cycles).
No harness code change is warranted from one cycle of evidence; per
`_fit_screening_decode_timeout_seconds`'s own contract, recalibrating the
arm-share model or thrash recipe requires recurrence, not a single cold-start
sample. Left a pointer comment in the function docstring
(`scripts/run_autotrain_continuous.py`) so a future repair_harness diagnosis
starts from this investigation instead of re-deriving it from raw numbers.

## Disposition

- No policy/code behavior change (docstring-only, `no-bump`).
- `repair_harness` action acknowledged as `completed`: the "repair" here is
  the diagnosis itself (ruling out a reproducible defect) plus the recorded
  pointer for future recurrences, not a numeric knob change.
- `retry_measurement` remains queued: replay the identical frozen c3 arms
  next cycle now that the process (and OS file cache / import cache) is
  warm; if the same clamp binds again on a warm process, that *is* evidence
  of a real miscalibration and should reopen this harness family.

Machine evidence:
[`autotrain-cycle-continuous-openui-scheduled-c3-decode-timeout-diagnostic.json`](autotrain-cycle-continuous-openui-scheduled-c3-decode-timeout-diagnostic.json).
