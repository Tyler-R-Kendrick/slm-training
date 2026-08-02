# Continuous autotrain cycle 6 results (2026-08-02, loop `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c6` |
| Cycle intent | `screening` — first size-matched test of the `component-edge` hypothesis (c5's rank-1 next-run priority; `component-plan` is rejected for this recipe) |
| Upstream / integration | `b8188a49` / `ada0f314` |
| Device | CPU |
| Steps | 20 / seed 100006 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke`, `--ship-gates` |
| Wall cap | 3 minutes |
| Hypothesis | `component_edge_loss_weight=1.0` (structural component-edge auxiliary supervision) improves smoke `structural_similarity` without lowering `parse_rate` |
| Primary metric | `smoke.structural_similarity` (direction: increase, minimum_effect 0.01) |

## Why this cycle exists

Cycle 5 ([results](continuous-openui-local-20260802-c5-results.md)) confirmed
`component-plan` (`component_plan_loss_weight=1.0`,
`structural_aux_head_profile="component-plan"`) is a genuine quality
regression for this recipe and ranked (0.90 confidence) testing the distinct,
size-matched `component-edge` hypothesis next. This cycle is that screen: a
fresh control/candidate pair, both trained for 20 steps from scratch (not a
frozen replay), differing only in `component_edge_loss_weight` (0.0 vs 1.0).

## Run matrix

| Arm | Levers | Params | smoke n | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | component_type_recall | latency_ms_p50 | Gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c6-control | component-edge off | 1,766,987 | 3 | 1.0 | 0.0 | 0.0964 | 0.8222 | 0.0833 | 1451.38 | **fail** (insufficient_n, quality thresholds) |
| c6-component-edge | component-edge **on** | 1,766,987 | 3 | 1.0 | 0.0 | 0.0964 | 0.8222 | 0.0833 | 1825.79 | **fail** (same) |

`measurement_complete: true` — both arms have a full AgentV ship-gate
scoreboard (`gates.pass=false`, `held_out`/`adversarial`/`ood`/`rico_held`
suites `missing_suite` at smoke scope, as expected).

Both arms trained cleanly with distinct `config_sha256` values (confirmed via
manifest diff, not a harness bug): the candidate applies
`component_edge_loss_weight=1.0`, the control keeps it at `0.0`. Training loss
diverges as expected from the added auxiliary loss term (`last_loss` 13.7197
control vs 14.8031 candidate — candidate loss is *higher* because it is now
optimizing an extra term, not because the primary objective degraded).

**Every decode-facing smoke metric is bit-identical between the two arms:**
`parse_rate`, `meaningful_program_rate`, `structural_similarity`,
`binder_reference_f1`, `component_type_recall`, `ast_beq_rate`,
`canonical_beq_rate`, `placeholder_fidelity`, and `reward_score` all match
exactly. Only `latency_ms_p50` differs (1451.38 → 1825.79 ms, candidate
slower — plausibly the extra aux-head forward pass). Primary delta
(component-edge − control) `structural_similarity`: **0.0** exactly.

## SDLC Phase A classification

`positive: false`, `stack_layer: false`, `action: no_stack_layer_non_positive`.

Reasons (from `sdlc_delivery.json`):

1. `fixture_insufficient_n:c6-component-edge` (n=3 < 20)
2. `fixture_insufficient_n:c6-control` (n=3 < 20)
3. `primary_metric_null_or_worse:smoke.structural_similarity:control=0.0964 candidate=0.0964 improvement=0.0`
4. `fixture_insufficient_n_alone`

This is a **genuine null result**, not a regression (unlike cycle 5's
`component-plan`, which moved `structural_similarity` the wrong direction and
also failed a non-regression gate on `binder_reference_f1`). At n=3
smoke-fixture scale and 20 training steps, `component_edge_loss_weight=1.0`
produced **zero measurable change** in any generated-output quality metric —
the hypothesis is neither confirmed nor rejected here, only shown to have no
detectable effect at this size/step budget. Ship gates correctly fail closed
on fixture n=3 (<20) and every quality threshold for both arms; no gate was
weakened to reach this result.

## Next-run priorities

1. **model:** `component-edge` is exhausted (as a completed non-positive
   screen) at `steps=20`/`n=3`; a bit-identical decode output across two
   differently-trained checkpoints suggests this fixture scale is
   under-powered to detect anything but a large effect. Test either a
   materially different recipe for `component-edge` (higher loss weight or
   step budget) or rotate to an untested lever from the screening bank rather
   than reselecting an already-completed arm.
2. **evaluation:** keep the matched control as the size-matched baseline every
   cycle.
3. **infrastructure:** soft ship-gate fails on fixture `n` never stop the
   continuous loop.
4. **model_build:** consider raising `steps` or fixture `n` for the next
   component-edge/component-plan retry so a real effect (if any) can be
   distinguished from fixture-scale noise.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c6/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c6-{control,component-edge}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-20260802-c6-results.json`
- Predecessor: [cycle 5 results](continuous-openui-local-20260802-c5-results.md) (`component-plan` rejected)
