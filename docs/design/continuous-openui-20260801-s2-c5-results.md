# Continuous autotrain cycle 5, second session (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` (second session; see [summary](continuous-openui-20260801-s2-summary.md)) |
| Campaign | `continuous-loop-20260801-c5` |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` (`smoke` suite, n=3) |
| Primary metric | `smoke.latency_ms_p50`, direction decrease |

## Run matrix

| Arm | Levers | n | completed | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c5-control | batch_size=2 | 3 | 2/3 | 1.0 | 0.0 | 0.05625 | 3888.95 |
| c5-batch1 | batch_size=1 | 3 | 1/3 | 1.0 | 0.0 | 0.0184 | 22880.57 |

Primary delta (batch1 − control): **+18991.62ms** (much worse).

## Diagnostics

1. Dropping `batch_size` from 2 to 1 made smoke p50 latency nearly 6x worse
   (3888.95ms → 22880.57ms) with no quality offset: `meaningful_program_rate`
   stayed 0.0 on both arms and `structural_similarity` fell (0.05625 →
   0.0184).
2. SDLC Phase A classifies **non-positive**
   (`primary_metric_null_or_worse`). Both arms also hit
   `fixture_insufficient_n` (n=3 vs the ≥20 floor), and `c5-batch1` only
   completed 1/3 smoke documents inside the wall cap, so this single cycle
   is not a precise measurement of `batch_size=1`'s effect size -- but the
   direction (slower, not better) is unambiguous enough to deprioritize it.

## SDLC Phase A

`positive=False`, `stack_layer=False`, `action=no_stack_layer_non_positive`.
Docs-only, local commit.

## Next-run priorities

1. Do not pursue `batch_size=1`; `batch_size=2` (current default) wins on
   every measured dimension this cycle.
2. If batch-size sensitivity is worth revisiting, test `batch_size>2`
   instead, at a larger `n`.
