# Continuous autotrain cycle 4 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260802` |
| Campaign | `continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c4` |
| Source | `b025766` |
| Device | CPU |
| Steps | 20 |
| Intent | `retry_measurement` (replay of the c3 frozen pair) |

## Run matrix

| Arm | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| c4-control | 3 | 1.0 | 0.333 | 0.231 | 4552.29 | eval completed; ship gates fail (insufficient n) |
| c4-canvas | 3 | 1.0 | 0.333 | 0.231 | 4107.90 | eval completed; ship gates fail (insufficient n) |

## What this confirms

This is a retry of the c3 frozen control/canvas pair. The c3-control decode timeout **did not
reproduce** — both arms completed with real scoreboards, so the timeout was a one-run timing
artifact rather than a persistent infrastructure regression.

The driver classified this cycle **positive** (`efficiency_win:mpr_per_ms` gain of ~10.8% with
quality held, `structural_similarity` unchanged) but explicitly **skipped stacking**:
`stack_action=positive_no_tracked_delta_skip_stack` — "metric win recorded; no code/docs delta —
skip stack PR; continue loop". Correct: no lever or code changed between c3 and this retry, so
there is nothing new to ship; the latency delta is retry-to-retry timing noise, not an
attributable improvement.

## Next-run priorities

1. **model:** the ranked matrix's next hypothesis is the size-matched `component-plan` lever
   (`c4-component-plan`).
2. **evaluation:** keep the matched control as the size-matched baseline every cycle.
3. **harness (separate):** dedicated repair pass for the 64 pre-existing `tests/test_evals`
   failures on `main` flagged in cycle 2's doc.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c4/`
- JSON twin: `continuous-openui-20260802-c4-results.json`
