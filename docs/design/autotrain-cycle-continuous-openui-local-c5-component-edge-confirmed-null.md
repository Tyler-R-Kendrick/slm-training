# Autotrain `continuous-openui-local` c5: component-edge confirmed, quality-tie null

**Verdict:** frozen replay of the [c4 control/component-edge
pair](autotrain-cycle-continuous-openui-local-c4-component-edge-control-timeout.md)
now **completes on both arms** with an exact quality tie — confirming c4's
control decode-timeout was a one-run CPU timing artifact, not a reproducible
executable blocker.

## Result matrix

| Arm | n | parse | meaningful | structure (primary) | binder F1 | component recall | p50 ms | decode timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 3 | 1.0 | 0.3333 | 0.4167 | 0.9524 | 0.25 | 23,578.00 | 0 |
| component-edge | 3 | 1.0 | 0.3333 | 0.4167 | 0.9524 | 0.25 | 23,620.00 | 0 |

Both arms are quality-identical; `smoke.structural_similarity` improvement is
exactly `0.0`. No harness lane was opened for the c4 timeout — it did not
reproduce.

## SDLC Phase A classification

`SDLC_PHASE_A NON_POSITIVE` — `fixture_insufficient_n` on both arms +
`primary_metric_null_or_worse` (improvement 0.0). **No stack layer opened.**
This closes the c4 `retry_measurement` action: the identical frozen pair
(`replay_of_manifest_sha256 912b622ccda9e867a37fc5bb6ccc4c0d50d3fbfd80a5404691dbbd50759a8fa0`)
now has a complete, usable scoreboard.

## Next-run priorities

1. **model** — this exact arm pair is exhausted (quality-tie null); test the
   distinct size-matched `component-plan` quality hypothesis next.
2. **evaluation** — keep the matched control every cycle.

No checkpoint was created or promoted, so no `MODEL_CARD.md` / README update is
required beyond the existing c4 roster row (already flagged as
measurement-incomplete/queued-for-replay; this cycle resolves that queue item
without changing the roster claim). Machine-readable values are in
[`autotrain-cycle-continuous-openui-local-c5-component-edge-confirmed-null.json`](autotrain-cycle-continuous-openui-local-c5-component-edge-confirmed-null.json).
