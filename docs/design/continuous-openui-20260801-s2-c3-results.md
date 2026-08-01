# Continuous autotrain cycle 3, second session (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` (second session; see [summary](continuous-openui-20260801-s2-summary.md)) |
| Campaign | `continuous-loop-20260801-c3` |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` (`smoke` suite, n=3) |
| Primary metric | `smoke.latency_ms_p50`, direction decrease |

## Run matrix

| Arm | Levers | n | completed | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c3-control | (none) | 3 | 3/3 | 1.0 | 0.0 | 0.1725 | 9589.97 |
| c3-both | grammar_completion_bounds + compact_active_canvas | 3 | 3/3 | 1.0 | 0.0 | 0.1725 | 9430.79 |

Primary delta (both − control): **-159.18ms** (a latency win).

## Diagnostics

1. Combining `grammar_completion_bounds` and `compact_active_canvas` shaved
   159ms off smoke p50 latency, but `meaningful_program_rate` stayed 0.0 on
   both arms and `structural_similarity` was identically 0.1725 -- no quality
   signal moved at all.
2. Per the quality-aware tradeoff policy, a pure latency delta with
   `mpr < 1/3` (the smoke n=3 floor) is **not** a free win --
   `latency_win_rejected_low_mpr`. SDLC Phase A classifies this cycle
   **non-positive**.

## SDLC Phase A

`positive=False`, `stack_layer=False`, `action=no_stack_layer_non_positive`.
Docs-only, local commit.

## Next-run priorities

1. Do not prioritize a dedicated matrix for `grammar_completion_bounds` +
   `compact_active_canvas` until a lever moves `meaningful_program_rate` or
   `structural_similarity` off the floor on the same suite -- this pairing is
   latency-only at fixture scale.
