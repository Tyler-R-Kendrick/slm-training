# Continuous autotrain: 2026-08-04 cycle 2 — infrastructure failure (decode timeout)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `336bec26` (on top of `main` tip `170eb5cf`, which
already includes #1431/#1432)

**Verdict:** infrastructure failure, not scoreable. Not a model result.

The `control` arm (1,755,764 matched params, `wf_smoke_v2`, `steps=20`, seed
100002) hit `decode_timeout_count=3` (exit code `124`) and never produced a
scoreboard; the `component-plan` candidate consequently never executed (exit
code `2`, `missing_scoreboard`). Neither arm produced usable evidence — no
`parse_rate`, `structural_similarity`, or other quality metric is available
for this cycle.

## Relation to prior evidence

This is the same soft-wall-clock decode-timeout class already diagnosed in
[`decode-timeout-hang-seed44-steps72-finding.md`](decode-timeout-hang-seed44-steps72-finding.md):
`_generate_chunk`'s `decode_timeout_seconds` guard
(`signal.setitimer(signal.ITIMER_REAL, ...)` + `SIGALRM`) is a well-known
CPython soft limit — the signal is only delivered between bytecode
instructions, so a single long native decode call can overrun the configured
timeout with zero output. That finding was `n=1` isolated-record evidence;
this cycle reproduces the same failure class at the full-arm level (all 3
records timed out on `control`).

## SDLC Phase A

**Non-positive** (`harness_failure`, `measurement_incomplete`,
`primary_metric_unavailable`). No stack layer; local commit only, per `sdlc`
autotrain-iteration-delivery. The driver correctly did not treat this as a
negative model result.

## Next priorities (ranked by the driver)

1. `retry_measurement`: replay the exact frozen `c2` pair once to test
   whether the control-only decode timeout reproduces (confidence 0.95).
2. If the timeout reproduces deterministically, this graduates to a
   `repair_harness` signal against `model_build`'s eval-runner decode-timeout
   enforcement rather than another retry.
3. Do not attribute this cycle's non-completion to either model hypothesis —
   neither arm produced usable evidence.

Machine evidence:
[`continuous-openui-20260804-c2-results.json`](continuous-openui-20260804-c2-results.json).
