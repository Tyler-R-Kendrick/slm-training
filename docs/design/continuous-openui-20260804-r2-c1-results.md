# Continuous autotrain cycle 1 results (2026-08-04, `continuous-openui-local-r2`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local-r2` |
| Campaign | `continuous-loop-20260804-continuous-openui-local--84ea77f4-c1` |
| Source | `0abaf07f` (current `main`) |
| Train | `wf_smoke_v2`, 21 steps / batch 2 / seed 100001 |
| Eval | `e938_role_safe_all_targets_v2` |

## Context: why a new loop-id

This session's prior `continuous-openui-local` lineage (campaigns c1-c4,
PR #1322) was closed by the repo owner as superseded: a concurrent loop
instance (`continuous-openui-202608`, id `39ee9cf7`) landed a better version
of the same AgentV `NODE_OPTIONS` self-heal plus decode-timeout evidence in
#1429 / #1425, already merged to `main`. This branch was reset to `main`
(`0abaf07f`) and a fresh loop-id started to avoid re-litigating superseded
work.

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| control | bounds off | 3 | 1.0 | 0.0 | 0.0575 | 5061.57 |
| bounds | bounds **on** | 3 | 1.0 | 0.0 | 0.0575 | 4243.04 |

Primary metric delta (bounds − control) on `smoke.structural_similarity`:
**0.0** (exact tie), matching the PR #1322 finding on the same lever.

## Diagnostics

1. Both the AgentV `NODE_OPTIONS` fix and the decode-timeout self-heal (both
   merged from #1429) held: this cycle completed a full measurement on the
   first try, no crash, no timeout.
2. Latency this run reverses sign from the earlier PR #1322 measurement:
   here `bounds` is 16.2% **faster** (4243.04 vs 5061.57 ms), whereas the
   PR #1322 c1/c3 runs showed `bounds` 5.68% **slower**. Absolute latencies
   are also ~3x higher than those earlier runs (4200-5000ms vs
   1500-1600ms). Taken together this is CPU-sandbox timing noise, not an
   attributable lever effect — consistent with the `39ee9cf7` loop's own
   c4 conclusion that retry-to-retry latency deltas on this fixture aren't
   attributable without a code/lever change.

## Next-run priorities

1. Do not treat either run's bounds latency delta as a real effect; sign
   reversal across runs confirms sandbox timing noise dominates at this
   fixture scale.
2. Test the size-matched `component-plan` quality hypothesis next per the
   driver's ranked priorities.
3. Do not promote or ship either checkpoint.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260804-continuous-openui-local--84ea77f4-c1/`
- JSON twin: `continuous-openui-20260804-r2-c1-results.json`
