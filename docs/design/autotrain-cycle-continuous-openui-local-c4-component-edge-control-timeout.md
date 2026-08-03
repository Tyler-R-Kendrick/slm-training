# Autotrain `continuous-openui-local` c4: component-edge completes, control decode-timeout

**Verdict:** fixture measurement **incomplete** — the `component-edge`
candidate produced the strongest raw quality numbers seen anywhere in this
loop so far, but the matched `control` cannot be attributed against it: all
3 smoke documents hit the typed 8s per-document decode timeout before the
control ever produced a scoreboard.

## Result matrix

| Arm | n scored | parse | meaningful | structure | binder F1 | p50 ms | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| component-edge | 3 | 1.0 | 0.3333 | 0.4167 | 0.9524 | 23,366.00 | complete (fails smoke-n / suite-coverage gates, not quality) |
| control | 0 | — | — | — | — | — | `decode_timeout_count=3`, no scoreboard |

Both arms are size-matched at 1,766,987 trainable params. The candidate's
numbers are informative but **not comparative** — `primary_metric_unavailable`
because the control has nothing to compare against.

## SDLC Phase A classification

`SDLC_PHASE_A NON_POSITIVE` — `measurement_incomplete` +
`harness_failure:experiment_failed` on the control, alongside
`executable_unblock:candidate_completed_after_control_error` (the candidate's
own completion is not itself blocked by anything, it simply has no partner).
**No stack layer opened.**

## Next-run priority (rank 1, confidence 0.95)

> The tail-supervised candidate completed while the matched control entered a
> typed decode timeout; replay the exact frozen pair once to test whether the
> runtime unblock reproduces.

This distinguishes a one-run CPU timing artifact (8s per-document budget is
marginal on this loop's shared CPU host) from a reproducible executable
blocker that would need a harness lane. The frozen manifest digest
(`912b622ccda9e867a37fc5bb6ccc4c0d50d3fbfd80a5404691dbbd50759a8fa0`) is
preserved for the exact replay; `retry_measurement` is queued as a
driver-owned execution action (not a predecessor-gating prerequisite).

No checkpoint was promoted (both are local-only, no-sync fixture
checkpoints), so no README summary change beyond the roster row is required.
Machine-readable values are in
[`autotrain-cycle-continuous-openui-local-c4-component-edge-control-timeout.json`](autotrain-cycle-continuous-openui-local-c4-component-edge-control-timeout.json).
