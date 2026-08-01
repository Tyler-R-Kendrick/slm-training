# Continuous autotrain cycles 1-2, third session (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` (third independent container session, run against [#1253](https://github.com/Tyler-R-Kendrick/slm-training/pull/1253)'s partial-suite-completion gate fix) |
| Campaigns | `continuous-loop-20260801-c1`, `continuous-loop-20260801-c2` |
| Source | `9c4bb5a305854c816690a936ddaad1f9f13f77f2` |
| Device | CPU |
| Eval | `e938_role_safe_all_targets_v2` (`smoke` suite, n=3) |

## Run matrix

| Cycle | Arm | Lever | completed | parse_rate | mpr | structural_similarity | latency_ms_p50 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | c1-control | (none) | 3/3 | 1.0 | 0.0 | 0.0575 | 2738.75 |
| 1 | c1-bounds | grammar_completion_bounds | 3/3 | 1.0 | 0.0 | 0.0575 | 2773.44 |
| 2 | c2-control | (none) | 3/3 | 1.0 | 0.0 | 0.32667 | 23047.69 |
| 2 | c2-canvas | compact_active_canvas | **2/3** | 1.0 | 0.0 | 0.2275 | 19846.39 |

## Diagnostics

1. **c1** isolates `grammar_completion_bounds`: latency got slightly worse
   (2738.75ms → 2773.44ms), `structural_similarity` unchanged.
   `primary_metric_null_or_worse` — non-positive. Consistent with session 2's
   [c6 finding](continuous-openui-20260801-s2-c6c7-results.md).
2. **c2** isolates `compact_active_canvas`: the naive latency delta
   (23047.69ms → 19846.39ms) looks like a large win, but `c2-canvas` only
   completed **2 of 3** smoke documents inside the wall cap while `c2-control`
   completed all 3. This is exactly the completion-race class documented in
   [`model-build-partial-suite-completion-gate-20260801.md`](model-build-partial-suite-completion-gate-20260801.md)
   (PR [#1253](https://github.com/Tyler-R-Kendrick/slm-training/pull/1253)) —
   the driver correctly emitted
   `primary_metric_incomparable_partial_suite:smoke.latency_ms_p50:control_completed=3.0/3.0:candidate_completed=2.0/3.0`
   instead of scoring a win.

This is a live confirmation that the completion gate fires correctly on a
real cycle, not just the synthetic regression tests added in #1253.

## SDLC Phase A

Both cycles: `positive=False`, `stack_layer=False`,
`action=no_stack_layer_non_positive`. Docs-only, local commit — no new
stacked layer.

## Next-run priorities

1. `grammar_completion_bounds` alone still shows no latency or quality
   benefit at this fixture scale.
2. Re-run `compact_active_canvas` once the underlying wall-clock completion
   race is addressed (larger wall budget or smaller `n`), since the gate now
   correctly withholds a verdict on mismatched completion instead of scoring
   a spurious win.
