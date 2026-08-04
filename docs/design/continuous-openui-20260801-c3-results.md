# Continuous autotrain cycle 3 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c3` |
| Source | `c1c4eca349b66f05684975575a3640ced50051ea` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | latency_ms_p50 | parse_rate | Status |
| --- | --- | ---: | ---: | --- |
| c3-control | both off | 7111.38 | 1.0 | eval completed; ship gates fail (insufficient n) |
| c3-both | `grammar_completion_bounds=true`, `compact_active_canvas=true` | 7340.17 | 1.0 | eval completed; ship gates fail (same) |

Primary delta (both − control) p50 latency: **+228.79 ms** (candidate slower).

## Diagnostics

1. Combining both levers regresses latency, consistent with each lever
   screened alone: `bounds` alone was +89.01ms worse (c1), `canvas` alone was
   +362.33ms worse (c2), and `both` together is +228.79ms worse (this cycle).
2. `parse_rate` and `meaningful_program_rate` are unchanged across all three
   screens — the regression is pure latency, not a quality/latency tradeoff.
3. Ship gates fail on fixture `insufficient_n` as expected at this scale.

## Next-run priorities

1. **model:** deprioritize `grammar_completion_bounds` and
   `compact_active_canvas` for this recipe — three independent screens (c1,
   c2, c3) all show a latency regression, no quality offset.
2. **model:** screen the remaining hypothesis-matrix candidates from this
   loop (`steps`, `batch1`) instead.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c3/`
- JSON twin: `continuous-openui-20260801-c3-results.json`
