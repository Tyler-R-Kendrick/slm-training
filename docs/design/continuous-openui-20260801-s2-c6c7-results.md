# Continuous autotrain cycles 6-7, second session (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` (second session; see [summary](continuous-openui-20260801-s2-summary.md)) |
| Campaigns | `continuous-loop-20260801-c6`, `continuous-loop-20260801-c7` |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Eval | `e938_role_safe_all_targets_v2` (`smoke` suite, n=3) |

## Run matrix

| Cycle | Arm | Lever | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 6 | c6-control | (none) | 1.0 | 0.0 | 0.0964 | 3212.18 |
| 6 | c6-bounds | grammar_completion_bounds | 1.0 | 0.0 | 0.0964 | 3251.19 |
| 7 | c7-control | (none) | 1.0 | 0.0 | 0.0575 | 3830.66 |
| 7 | c7-canvas | compact_active_canvas | 1.0 | 0.0 | 0.0575 | 3781.32 |

## Diagnostics

1. **c6** isolates `grammar_completion_bounds`: latency got 39ms *worse*
   (3212.18ms → 3251.19ms), structural_similarity unchanged.
   `primary_metric_null_or_worse` — non-positive.
2. **c7** isolates `compact_active_canvas`: latency improved 49ms
   (3830.66ms → 3781.32ms), but structural_similarity was identical on both
   arms and `meaningful_program_rate` stayed 0.0.
   `latency_win_rejected_low_mpr` — non-positive.
3. Cross-reference with
   [cycle 3](continuous-openui-20260801-s2-c3-results.md), which combined
   both levers and saw a 159ms latency win with no quality movement: isolating
   the two levers here shows `compact_active_canvas` (not
   `grammar_completion_bounds`) is the one carrying that latency effect —
   `grammar_completion_bounds` alone is a small regression.

## SDLC Phase A

Both cycles: `positive=False`, `stack_layer=False`,
`action=no_stack_layer_non_positive`. Docs-only, local commit.

## Next-run priorities

1. Do not prioritize `grammar_completion_bounds` alone — no quality or
   latency benefit measured in isolation.
2. `compact_active_canvas` in isolation reproduces roughly cycle 3's
   latency-only improvement; still rejected as a free win at `mpr=0.0`. Not
   worth a dedicated matrix unless a run shows it moving quality.
