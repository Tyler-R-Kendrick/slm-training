# Continuous autotrain cycle 7 results (2026-08-02, loop `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c7` |
| Cycle intent | `screening` — driver's hypothesizer re-selected `component-plan` (rank-1 candidate at cycle 6's close was `component-inventory`, but the matrix re-proposed `component-plan` as this cycle's `recommended_experiment_id`) |
| Upstream / integration | `b8188a49` / `4f64a4b1` |
| Device | CPU |
| Steps | 20 (recorded 21) / seed 100007 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke`, `--ship-gates` |
| Wall cap | 3 minutes |
| Hypothesis | `component_plan_loss_weight=1.0` (structural component-plan auxiliary supervision) improves smoke `structural_similarity` without lowering `parse_rate` |
| Primary metric | `smoke.structural_similarity` (direction: increase) |

## Why this cycle exists

The supervised driver's own hypothesizer/exhaustion-cooldown logic was left to
pick the next experiment (no forced hypothesis flag). It proposed
`component-plan` again — the same lever family cycle 5 confirmed as a genuine
quality regression (`structural_similarity` 0.2308→0.1725,
`binder_reference_f1` 0.7333→0.6333, via a frozen replay of cycle 3's
checkpoints). This cycle is **not** a replay: it is a fresh from-scratch
control/candidate pair at seed `100007`, a different seed/checkpoint lineage
than cycle 5's. Per the task contract ("if it picks an already-screened arm
anyway, that's fine — just run it and document honestly"), the run proceeded
as selected.

## Run matrix

| Arm | Levers | Params | smoke n | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency_ms_p50 | Gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c7-control | component-plan off | 1,755,764 | 3 | 1.0 | 0.0 | 0.0575 | 0.6333 | 1602.13 | **fail** (insufficient_n, quality thresholds) |
| c7-component-plan | component-plan **on** | 1,755,764 | 3 | 1.0 | 0.0 | 0.0575 | 0.6333 | 1383.81 | **fail** (same) |

`measurement_complete: true` — both arms have a full AgentV ship-gate
scoreboard (`gates.pass=false`, `held_out`/`adversarial`/`ood`/`rico_held`
suites `missing_suite` at smoke scope, as expected).

Both arms trained cleanly with distinct recipe values (confirmed via
`train_summary.json` diff, not a harness bug): the candidate applies
`component_plan_loss_weight=1.0`, the control keeps it at `0.0`. Training loss
diverges as expected from the added auxiliary loss term (`last_loss` 15.1286
control vs 18.1728 candidate — candidate loss is *higher* because it is now
optimizing an extra term, not because the primary objective degraded).

**Every decode-facing smoke metric is bit-identical between the two arms:**
`parse_rate`, `meaningful_program_rate`, `structural_similarity`,
`binder_reference_f1`, `component_type_recall`, `ast_beq_rate`,
`canonical_beq_rate`, `placeholder_fidelity`, and `reward_score` all match
exactly. Only `latency_ms_p50` differs (1602.13 → 1383.81 ms, candidate
*faster* this time — the opposite direction from cycle 6's component-edge
latency delta, underscoring this is fixture-scale timing noise, not a real
speed effect). Primary delta (component-plan − control) `structural_similarity`:
**0.0** exactly.

## SDLC Phase A classification

`positive: false`, `stack_layer: false`, `action: no_stack_layer_non_positive`.

Reasons (from `sdlc_delivery.json`):

1. `fixture_insufficient_n:c7-component-plan` (n=3 < 20)
2. `fixture_insufficient_n:c7-control` (n=3 < 20)
3. `primary_metric_null_or_worse:smoke.structural_similarity:control=0.0575 candidate=0.0575 improvement=0.0`
4. `fixture_insufficient_n_alone`

This is a **genuine null result at this seed**, not a reproduction of cycle
5's measured regression — cycle 5 used a different lineage (evaluation-only
frozen replay of cycle 3's already-trained checkpoints), while this cycle
trained both arms from scratch at a new seed. The null does **not** overturn
cycle 5's confirmed rejection; it shows the component-plan effect (when
present) is seed/lineage-sensitive and not reliably reproducible as a
detectable signal at n=3/steps=20 fixture scale. Ship gates correctly fail
closed on fixture n=3 (<20) and every quality threshold for both arms; no gate
was weakened to reach this result.

## Next-run priorities

1. **model:** the completed non-positive component-plan arm is exhausted
   again; test the distinct size-matched `component-inventory` quality
   hypothesis next (rank 1, confidence 0.90) — component-plan and
   component-edge are both now spent for `steps=20`/`n=3`.
2. **evaluation:** keep the matched control as the size-matched baseline every
   cycle.
3. **model:** rotate thrash recommendation across the lever bank (not
   bounds-only) — the completed candidate is exhausted and cannot be selected
   again without a new preregistered hypothesis.
4. **infrastructure:** soft ship-gate fails on fixture `n` never stop the
   continuous loop.
5. **model_build:** confirmed champions promote under cadence; thrash only
   screens.

## Screening-bank assessment

Cycles 3/5 (component-plan, rejected — genuine regression), 6
(component-edge, null), and 7 (component-plan re-screen, null) are all
non-positive at this `steps=20`/`n=3` fixture scale. Three-plus consecutive
non-positive cycles at the same fixture size is the explicit signal the task
contract asks to surface plainly: the screening bank at this fixture scale
looks exhausted for producing a *positive* result, though it continues to
produce genuine (non-null) information — cycle 5's regression is a real,
reproducible-in-direction finding, distinct from cycles 6/7's bit-identical
nulls. The orchestrating session should weigh whether to keep spending cycles
screening individual aux-loss levers at n=3/steps=20, or escalate to a larger
fixture (`n`, `steps`) so a true effect (if any) is distinguishable from
fixture-scale noise, per this doc's and cycle 6's shared observation that
bit-identical decode output across differently-trained checkpoints suggests
under-powered measurement.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c7/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c7-{control,component-plan}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-20260802-c7-results.json`
- Predecessor: [cycle 6 results](continuous-openui-local-20260802-c6-results.md) (`component-edge` null screen)
