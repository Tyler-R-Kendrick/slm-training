# Autotrain c4 (continuous-openui-scheduled): decode-timeout confirmed reproducible — corrects c3

**Verdict:** measurement incomplete, infrastructure limitation — corrects the
c3 diagnostic's "cold sandbox" hypothesis with data. Not a harness defect to
patch; a genuine hardware-speed characteristic of this sandbox class.

[`autotrain-cycle-continuous-openui-scheduled-c3-decode-timeout-diagnostic.md`](autotrain-cycle-continuous-openui-scheduled-c3-decode-timeout-diagnostic.md)
hypothesized c3's `compiler_ms_mean≈23,200` (vs an ~8.0s fitted budget) was a
cold-sandbox artifact (fresh `.venv`/`npm ci` moments earlier) and recommended
a warm retry. c4 replayed the identical frozen arm
(`00574c47f6362eaae01b999e28683e721f092026d1c561e5669ef22a0a811210`) with a
warm process (no new installs, prior cycle's imports/caches hot):
`compiler_ms_mean=23216.8` (control) / `23100.9` (canvas) — within 1% of c3's
`23165.5` / `23242.0`. `/proc/loadavg` at the time of c4 was `0.56 1.15 1.29`
on a 4-vCPU sandbox (`nproc`), not contended.

## Corrected diagnosis

Two consecutive cycles with near-identical decode latency, on an uncontended
host, rules out a transient cold-start explanation. This sandbox's CPU
genuinely takes ~23s to run the compiler-tree search this repo's historical
smoke decode completes in ~0.9-4.0s elsewhere. The fitted
`screening_decode_timeout_seconds` (~8.0s, `arm_wall_seconds≈52.76`,
`min_train_floor_seconds=20`, `eval_overhead_seconds=8`, `smoke_n=3`) cannot
be raised to cover 23s/record without exceeding the arm's wall share
(`3×23s + 20 + 8 = 97s > 52.76s` — the arm wall itself is too small for this
hardware's decode speed, not just the timeout knob).

This is **not** treated as a global policy miscalibration: the 8s default and
`<=12.0s` clamp ceiling (`test_fit_screening_decode_fits_arm_wall`) are tuned
against this repo's broad historical decode-latency evidence across many
prior sandboxes, where the clamp does not bind. Changing the global default
from one sandbox class's measurement risks misconfiguring environments where
decode is fast. No policy/code change made from this evidence.

## Disposition

- `retry_measurement` is not queued again: a third identical retry on this
  sandbox is expected to reproduce the same timeout (predictable from the
  arm-wall arithmetic above), which is exactly the loop law's
  same-blocker-repeats condition — retrying again burns a cycle without new
  information.
- This loop's honest infrastructure repairs this session (AgentV SDK
  self-heal, design_md bridge NODE_OPTIONS fix, `dsl.analysis.arity` torch-leak
  fix, `setup_dev_env.sh` completion — commits `1c48ac9`, `2aedec9`,
  `3ce1b58`) are real and verified, but none produced a **positive** result
  under `sdlc` autotrain-iteration-delivery's gate (primary-metric win,
  ship-quality win, or an identical-arm replay reaching a usable scoreboard):
  every replay still ends at `measurement_incomplete`. No stack layer opens
  for this iteration; local commits stand as the record.
- A future cycle on faster (or GPU-backed) compute, or a deliberate,
  separately-reviewed thrash-recipe recalibration (smaller `smoke_n`, fewer
  `steps`, or a larger screening wall share) backed by measurements across
  multiple sandbox classes, can retry this arm honestly.

Machine evidence:
[`autotrain-cycle-continuous-openui-scheduled-c3-decode-timeout-diagnostic.json`](autotrain-cycle-continuous-openui-scheduled-c3-decode-timeout-diagnostic.json)
(c3) and cycle c4's campaign under
`outputs/autoresearch/continuous-loop-20260804-continuous-openui-schedu-1e62ecf9-c4/`
(local, explicit no-sync, not reusable/promoted/ship).
