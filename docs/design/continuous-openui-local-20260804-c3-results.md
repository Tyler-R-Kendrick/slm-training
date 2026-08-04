# Continuous autotrain cycle 3 results (2026-08-04)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c3` |
| Cycle intent | `screening` (component-plan-decode weighting hypothesis) |
| Source | `eba6db3044076285581b80cfe5294a2ecbcee8a1` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |
| Trainable params | 1,755,764 (both arms, size-matched) |

## Run matrix

| Arm | `structural_similarity` | `meaningful_program_rate` | `binder_reference_f1` | `latency_ms_p50` |
| --- | --- | --- | --- | --- |
| c3-control | 0.2308 | 0.3333 | 0.7333 | 12,685.96 |
| c3-component-plan | 0.1725 | 0.0 | 0.6333 | 11,866.74 |

Primary metric delta: **-0.0583** (regression). `binder_reference_f1` also regresses
(0.7333 -> 0.6333), tripping the non-regression guard.

## Outcome

The component-plan-decode candidate regresses the primary metric and a guarded
non-regression metric against its size-matched control. Both arms still fail the smoke ship
gate on fixture-scale `insufficient_n` (`n=3`, need `>=20`).

## SDLC Phase A classification

**Non-positive**: primary metric moved in the harmful direction
(`primary_metric_null_or_worse`) and a guarded metric regressed
(`non_regression_fail:binder_reference_f1`). No stacked PR opens for this cycle. Local commit +
docs only.

## Next-run priorities

1. **model:** the completed non-positive arm is exhausted; test the distinct size-matched
   `component-edge` quality hypothesis next (confidence 0.90).
2. **evaluation:** keep the matched control as the size-matched baseline every cycle
   (confidence 0.70).

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260804-continuous-openui-local-8c0b60dd-c3/`
- Runs: `.../runs/c20260804-continuous-openui-local-8c0b60dd-c3-control/`,
  `.../runs/c20260804-continuous-openui-local-8c0b60dd-c3-component-plan/`
- JSON twin: `continuous-openui-local-20260804-c3-results.json`
