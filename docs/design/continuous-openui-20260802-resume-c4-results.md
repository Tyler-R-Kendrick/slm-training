# Continuous autotrain resumed-session cycle 4 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

> See [`continuous-openui-20260802-resume-c2-results.md`](continuous-openui-20260802-resume-c2-results.md)
> for why this resumed session's `campaign_id` namespace collides with the earlier same-day
> `continuous-openui-20260802-c4-results.md`. This doc covers a distinct campaign run.

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260802` |
| Campaign | `continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c4` |
| Source | `ef4bc4e9a4f260ca474d410a31221e72a9b1a1fa` |
| Device | CPU |
| Steps | 20 |
| Params (both arms) | 1,766,992 |
| Lever tested | `component-edge` |

## Run matrix

| Arm | smoke n | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency_ms_p50 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c4-control | 3 | — | — | — | — | — | **measurement_incomplete** (`decode_timeout_count=3`, every metric `None`) |
| c4-component-edge | 3 | 1.0 | 0.333 | 0.4167 | 0.952 | 7703.69 | eval completed; ship gates fail (insufficient n) |

## What this confirms

Same pattern already observed in the earlier same-day session's c3→c4 cycle: the **control** arm
hit a typed decode timeout on all 3 smoke docs, so its scoreboard has every quality metric as
`None` — a `measurement_incomplete` harness failure, not a real control measurement. The
size-matched **component-edge** candidate completed normally with the strongest smoke numbers
seen across this session's cycles so far (`structural_similarity=0.4167`,
`binder_reference_f1=0.9524`, `meaningful_program_rate=0.333`). Without a valid matched control in
the same cycle, `primary_metric` is unavailable and the cycle is **inconclusive, not positive** —
the driver correctly emitted `executable_unblock:candidate_completed_after_control_error` and
classified this non-positive (no stack layer) pending a retry.

## Next-run priorities

1. **infrastructure:** `retry_measurement` — replay the exact frozen `c4-control` /
   `c4-component-edge` pair once to test whether the control-only decode timeout reproduces
   (queued).
2. **model:** `component-edge`'s strong smoke numbers are promising but unconfirmed until a valid
   matched control exists in the same cycle.
3. **infrastructure:** soft ship-gate fails and single-run decode timeouts never stop the
   continuous loop.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c4/`
- JSON twin: `continuous-openui-20260802-resume-c4-results.json`
