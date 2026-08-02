# Continuous autotrain cycle 3 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Distinct hypothesis from the driver's cycle-2 priority: `component_plan`
lever instead of `grammar_completion_bounds`, against a fresh matched
control (params grew to 1.756e6 vs 1.609e6 in cycles 1-2).

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c3` |
| Source / integration | `b8188a49` / `74e9ca95` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c3-control | component_plan off | 3 | 1.0 | 0.333 | 0.2308 | 0.7333 | 2846.07 | **ship gates reject** |
| c3-component-plan | component_plan **on** | 3 | 1.0 | 0.0 | 0.1725 | 0.6333 | 3021.77 | **ship gates reject** |

Primary metric (`smoke.structural_similarity`) delta: **-0.0583** (candidate
worse than control).

## Diagnostics

1. `component_plan` **regresses** `structural_similarity`
   (0.2308 → 0.1725), `meaningful_program_rate` (0.333 → 0.0), and
   `binder_reference_f1` (0.7333 → 0.6333) vs its matched control at this
   size/step count — a non-regression check on `binder_reference_f1`
   explicitly failed.
2. Both arms still fail `insufficient_n` (3 < 20) — the `smoke` fixture is
   inherently too small to clear ship-quality gates regardless of lever
   choice; only `held_out`/`adversarial`/`ood`/`rico_held` suites (not run
   here) or a larger smoke fixture can clear that gate.
3. Latency also regressed slightly (2846ms → 3022ms p50), consistent with
   the quality regression rather than a speed/quality tradeoff.

## Classification

Non-positive (`SDLC_PHASE_A NON_POSITIVE`, `stack_layer=False`,
`action=no_stack_layer_non_positive`): `fixture_insufficient_n` on both
arms, `non_regression_fail:binder_reference_f1`, and
`primary_metric_null_or_worse` (delta -0.0583). No stacked PR opened.

## Next-run priorities (from driver)

1. **model:** `component_plan` is exhausted as a lever here; try the
   distinct size-matched `component-edge` hypothesis next (rank 1,
   confidence 0.90).
2. **evaluation:** keep the matched control as the size-matched baseline
   every cycle.
3. **model:** rotate thrash recommendation across the lever bank rather
   than repeating one lever.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c3/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c3-control/`,
  `.../runs/c20260802-continuous-openui-local-8c0b60dd-c3-component-plan/`
- JSON twin: `continuous-openui-local-20260802-c3-results.json`
