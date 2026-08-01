# Continuous autotrain cycles 5-6, third session (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` (third session; see [c3c4](continuous-openui-20260801-s3-c3c4-results.md)) |
| Campaigns | `continuous-loop-20260801-c5`, `continuous-loop-20260801-c6` |
| Source | `d99035b565ef8a8c6c8fb59f1c5949bd84fa482d` |
| Device | CPU |
| Eval | `e938_role_safe_all_targets_v2` (`smoke` suite, n=3) |

## Run matrix

| Cycle | Arm | Lever | completed | structural_similarity | latency_ms_p50 |
| --- | --- | --- | ---: | ---: | ---: |
| 5 | c5-control | (none) | 2/3 | 0.05625 | 3169.32 |
| 5 | c5-batch1 | batch_size=1 | 1/3 | 0.0184 | 17440.55 |
| 6 | c6-control | (none) | 3/3 | 0.0964 | 2457.93 |
| 6 | c6-bounds | grammar_completion_bounds | 3/3 | 0.0964 | 2472.98 |

## Diagnostics

1. **c5**: `batch_size=1` is unambiguously ~5.5x slower with no quality
   offset. Both arms also completed only part of the suite (2/3 and 1/3), so
   the partial-suite gate additionally flags
   `primary_metric_incomparable_partial_suite` — but the direction here is
   already unambiguous regardless of the completion mismatch.
   Non-positive, confirming [session 2's c5 finding](continuous-openui-20260801-s2-c5-results.md).
2. **c6**: `grammar_completion_bounds` alone is a small latency regression
   (2457.93ms → 2472.98ms), `structural_similarity` unchanged. Non-positive,
   confirming [session 2's c6 finding](continuous-openui-20260801-s2-c6c7-results.md).

## SDLC Phase A

Both cycles: `positive=False`, `stack_layer=False`,
`action=no_stack_layer_non_positive`. Docs-only, local commit.

## Next-run priorities

1. `batch_size=1` is confirmed non-viable across two independent sessions —
   deprioritize entirely.
2. `grammar_completion_bounds` alone is confirmed no-benefit across two
   independent sessions — deprioritize in isolation.
