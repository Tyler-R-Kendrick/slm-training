# Continuous autotrain cycles 7-8, third session (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` (third session; see [c5c6](continuous-openui-20260801-s3-c5c6-results.md)) |
| Campaigns | `continuous-loop-20260801-c7`, `continuous-loop-20260801-c8` |
| Source | `92a9cef7919c10b706d7eaebbd27b008d55bc291` |
| Device | CPU |

## Run matrix

| Cycle | Arm | Lever | `held_out` completed | `structural_similarity` | latency_ms_p50 |
| --- | --- | --- | ---: | ---: | ---: |
| 7 | c7-control | (none) | — (smoke only) | 0.0575 | 2816.05 |
| 7 | c7-canvas | compact_active_canvas | — (smoke only) | 0.0575 | 2761.13 |
| 8 | c8-control | (none) | 5/5 | 0.09758 | 2562.82 |
| 8 | c8-both | grammar_completion_bounds + compact_active_canvas | 5/5 | 0.09758 | 2510.26 |

## Diagnostics

1. **c7**: `compact_active_canvas` alone shaves ~55ms off smoke p50 latency,
   `structural_similarity` identical — `latency_win_rejected_low_mpr`,
   non-positive. Confirms this lever is latency-only across a third
   independent session now.
2. **c8**: promotion-role replay of the combined-lever recipe, this time with
   **both arms fully completing `held_out`** (5/5 each). `held_out.structural_similarity`
   is bit-identical (`0.09758`) between control and candidate — a clean zero
   effect, not a confound. This is the kind of trustworthy null result the
   partial-suite-completion gate (#1253) makes possible: no ambiguity from
   mismatched completion, just genuinely no measured quality effect at this
   recipe.

## SDLC Phase A

Both cycles: `positive=False`, `stack_layer=False`,
`action=no_stack_layer_non_positive`. Docs-only, local commit.

## Next-run priorities

1. `compact_active_canvas` confirmed latency-only (no quality movement)
   across three independent sessions now.
2. `grammar_completion_bounds` + `compact_active_canvas` combined show zero
   `held_out` effect when measurement is trustworthy — not worth a dedicated
   matrix on quality grounds.

## Session 3 wrap-up

Eight cycles this session, all non-positive by the SDLC Phase A gate — no
new stack layer beyond the harness fix itself
([#1253](https://github.com/Tyler-R-Kendrick/slm-training/pull/1253)). The
session's substantive contribution is the harness fix and the resulting
resolution of the cross-session steps-lever disagreement (see
[c3c4](continuous-openui-20260801-s3-c3c4-results.md)), not a new training
lever win. All lever findings this session (`grammar_completion_bounds`
alone: small regression; `compact_active_canvas` alone: latency-only;
`batch_size=1`: much worse; combined bounds+canvas: latency-only or zero
quality effect) corroborate session 2's independent findings rather than
contradicting them.
