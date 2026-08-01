# Continuous autotrain cycle — 2026-08-01, campaign `continuous-loop-20260801-c4`

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-c4` (cycle 4, predecessor `continuous-loop-20260801-c3`) |
| Role | **promotion** (first promotion-role cycle this session — queued champion confirmation) |
| Primary metric | `held_out.structural_similarity` (increase) |
| Device | CPU |
| Steps | 20 / batch 2 |
| Train | `wf_smoke_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | smoke mpr | smoke structural_similarity | held_out structural_similarity | held_out latency_ms_p50 | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c4-control | 0.333 | 0.417 | 0.38248 | 19175.31 | eval completed; gates fail |
| c20260801-c4-steps | 0.333 | 0.510 | 0.37006 | 9816.32 | eval completed; gates fail |

## SDLC Phase A classification

`positive: false`, `stack_layer: false` — **rejected on the promotion
metric**: `primary_metric_null_or_worse` (`held_out.structural_similarity`
0.38248 control vs 0.37006 candidate, a **regression**), plus
`fixture_insufficient_n` on both arms.

## Diagnostics

1. First cycle today where `smoke.meaningful_program_rate` is non-zero
   (0.333 on both arms) — the fixture is producing at least one meaningful
   program per suite now, unlike cycles c1–c3.
2. The `steps` candidate looks attractive on smoke-level signals alone
   (structural_similarity 0.51 vs 0.417, latency roughly halved), but the
   **held-out** suite — the actual promotion gate — shows a regression
   (0.370 vs 0.382). The promotion path correctly used `held_out`, not
   `smoke`, as the primary metric and blocked promotion.
3. This is a useful confirmation that the champion-promotion funnel fixed in
   #1242–#1245 is doing its job: a candidate that would look like a win on
   smoke-only screening is rejected once evaluated on held-out data.

## Next-run priorities

1. **model:** investigate why `smoke.structural_similarity` and
   `held_out.structural_similarity` move in opposite directions for the
   `steps` candidate — do not promote until reconciled.
2. **infrastructure:** none outstanding.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c4/`
- JSON twin: `continuous-openui-20260801-c4-results.json`
