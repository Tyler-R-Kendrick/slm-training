# Continuous autotrain cycle 8 results (2026-08-02, loop `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c8` |
| Cycle intent | `promotion` — driver's hypothesizer selected `component-inventory`, its own rank-1 recommendation from cycles 6 and 7, which it had twice recommended but not previously selected |
| Upstream / integration | `b8188a49` / `0044e56b` |
| Device | CPU |
| Steps | 20 requested (recorded 22) / seed 100008 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suites `smoke` + `held_out`, `--ship-gates` |
| Wall cap | 3 minutes |
| Hypothesis | `component_inventory_loss_weight=1.0` (structural component-inventory auxiliary head, `structural_aux_head_profile=component-inventory`) improves `held_out.structural_similarity` without lowering `parse_rate` |
| Primary metric | `held_out.structural_similarity` (direction: increase) |

## Why this cycle exists

The supervised driver's own hypothesizer was left to pick the next experiment
(no forced hypothesis flag, per task contract). Cycles 6 and 7 both closed
with `component-inventory` as the rank-1 speculative next-run priority
(confidence 0.90), but the driver's cooldown/dedup logic re-picked an
already-screened arm both times (`component-edge` in cycle 6 itself,
`component-plan` again in cycle 7). This cycle it actually selected
`component-inventory` — the first genuinely fresh-this-session arm run since
cycle 3. (Note: `component-inventory` was screened non-positive once before in
a much earlier, separate historical session/PR — #1301 — but the driver's
cross-session history did not flag it as exhausted here; it ran as a normal
fresh control/candidate pair.)

## Run matrix

| Arm | Levers | Params | smoke n | smoke structural_similarity | held_out n | held_out structural_similarity | latency_ms_p50 (smoke / held_out) | Gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c8-control | component-inventory off | 1,682,363 | 3 | 0.174167 | 5 | 0.09758 | 1189.96 / 1335.91 | **fail** (insufficient_n, quality thresholds) |
| c8-component-inventory | component-inventory **on** | 1,682,363 | 3 | 0.174167 | 5 | 0.09758 | 1246.61 / 1339.05 | **fail** (same) |

`measurement_complete: true` — both arms have a full AgentV ship-gate
scoreboard (`gates.pass=false`; `adversarial`/`ood`/`rico_held` suites
`missing_suite` at this fixture scope, as expected — this cycle ran both
`smoke` and `held_out`, one suite more than cycles 5–7).

Both arms trained cleanly with distinct recipe values (confirmed via
`train_summary.json` diff, not a harness bug): the candidate applies
`component_inventory_loss_weight=1.0`, the control keeps it at `0.0`.
Training loss diverges as expected from the added auxiliary loss term
(`last_loss` 19.9561 control vs 21.1865 candidate — candidate loss is
*higher* because it is now optimizing an extra term, not because the primary
objective degraded).

**Every decode-facing quality metric is bit-identical between the two arms on
both suites:** `parse_rate`, `meaningful_program_rate`,
`structural_similarity`, `binder_reference_f1`, `component_type_recall`,
`ast_beq_rate`, `canonical_beq_rate`, `placeholder_fidelity`, and
`reward_score` all match exactly on `smoke` and on `held_out`. Only
`latency_ms_p50` differs (smoke +56.65 ms, held_out +3.14 ms, candidate
slightly slower — consistent with fixture-scale timing noise, not a real
speed effect, and in the opposite direction of cycle 7's latency delta).
Primary metric (`held_out.structural_similarity`, candidate − control):
**0.0** exactly.

## SDLC Phase A classification

`positive: false`, `stack_layer: false`, `action: no_stack_layer_non_positive`.

Reasons (from `sdlc_delivery.json`):

1. `fixture_insufficient_n:c8-control` (smoke n=3 < 20, held_out n=5 < 20)
2. `fixture_insufficient_n:c8-component-inventory` (same)
3. `primary_metric_null_or_worse:held_out.structural_similarity:control=0.09758 candidate=0.09758 improvement=0.0`
4. `fixture_insufficient_n_alone`

This is a **bit-identical null result**, the same pattern cycle 6
(`component-edge`) showed — not a genuine regression like cycle 5's
`component-plan` measurement. Ship gates correctly fail closed on fixture
insufficient_n and every quality threshold for both arms and both suites; no
gate was weakened to reach this result.

## Next-run priorities

1. **model:** the completed non-positive component-inventory arm is exhausted;
   test the distinct size-matched `binder-topology` quality hypothesis next
   (rank 1, confidence 0.90) — component-plan, component-edge, and
   component-inventory are now all spent for `steps=20-22`/`n=3` (smoke) /
   `n=5` (held_out).
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

Cycles 3/5 (`component-plan`, rejected — genuine regression), 6
(`component-edge`, bit-identical null), 7 (`component-plan` re-screen,
bit-identical null), and now 8 (`component-inventory`, bit-identical null)
are all non-positive at this fixture scale. This is the **fifth consecutive
non-positive cycle** in this session (counting 1-2, 3/5, 6, 7, 8 as five
distinct screening instances). This cycle does not change the picture
established by cycles 6-7: it is *more of the same* — a third bit-identical
null across a third distinct structural aux-loss lever, at the same
`n=3`/`n=5`/`steps≈20` fixture scale, strongly reinforcing (not newly
discovering) that this fixture is under-powered to distinguish most of these
levers' effects from zero. Only cycle 5's `component-plan` measurement (a
frozen-replay comparison across a real harness fix boundary) has shown a
non-null, reproducible-in-direction signal so far this session; every
from-scratch same-seed control/candidate pair at this fixture scale
(cycles 6, 7, 8) has come back bit-identical on every quality metric. The
orchestrating session should weigh escalating fixture scale (`n`, `steps`)
before continuing to screen additional individual aux-loss levers one at a
time at `n=3`/`steps=20`, since three of three from-scratch same-seed
screens this session have been unable to detect any lever effect at all.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c8/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c8-{control,component-inventory}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-20260802-c8-results.json`
- Predecessor: [cycle 7 results](continuous-openui-local-20260802-c7-results.md) (`component-plan` re-screen, bit-identical null)
