# Continuous autotrain cycle 5 results (2026-08-02, loop `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c5` |
| Cycle intent | `retry_measurement` — frozen replay of cycle 3's control/component-plan arm |
| Upstream / integration | `b8188a49` / `56fb65a7` |
| Device | CPU |
| Steps | 20 (reused c3's completed train stage; evaluation-only replay, no new checkpoint) |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke`, `--ship-gates` |
| Wall cap | 3 minutes |

## Why this cycle exists

Cycle 3 ([results](continuous-openui-local-20260802-c3-results.md)) trained both
arms of the `component-plan` hypothesis but never produced a ship-gate
`scoreboard.json` for either arm — a recurrence of the cycle 1/2
`NODE_OPTIONS="--import tsx"` AgentV blocker, this time because the driver
invocation itself (not just `npm ci`) inherited the ambient env. The first
automatic `retry_measurement` attempt (would-be campaign `...-c4`) then crashed
on a genuine harness bug: `_apply_frozen_replay` recovered the screening-bank
arm slug via `rsplit("-", 1)[-1]`, which truncates any hyphenated slug —
`"...-c3-component-plan"` became `"plan"`, not a bank member, so the driver
raised `RuntimeError: unsupported automatic frozen replay arm: plan`. This
reproduced on the frozen input and named exactly one canonical family
(`autoresearch`), so it was routed through `improve-openui-harnesses`: fixed
in `scripts/run_autotrain_continuous.py` (commit `56fb65a7`) by recovering the
slug from the known source-campaign prefix (falling back to the longest
matching bank slug by suffix for older callers), with two new regression
tests and a `harness.autoresearch.experiment_campaign` v59→v60 bump. This
cycle (`c5`) is the successful replay: `NODE_OPTIONS` unset for the whole
process tree, identical frozen control/candidate arm, per the
`retry_measurement` handoff action and the continuous-loop frozen-replay
contract.

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency_ms_p50 | Gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c5-control | component-plan off | 3 | 1.0 | 0.3333 | 0.2308 | 0.7333 | 3486.09 | **fail** (insufficient_n, quality thresholds) |
| c5-component-plan | component-plan **on** | 3 | 1.0 | 0.0 | 0.1725 | 0.6333 | 3363.32 | **fail** (same, plus regression) |

`measurement_complete: true` — both arms now have a full AgentV ship-gate
scoreboard (`gates.pass=false`, `held_out`/`adversarial`/`ood`/`rico_held`
suites `missing_suite` at smoke scope, as expected).

Primary delta (component-plan − control) `structural_similarity`: **-0.0583**
(candidate worse — wrong direction for a metric whose declared direction is
increase). `binder_reference_f1` also regresses (0.7333 → 0.6333) and
`meaningful_program_rate` drops to **0.0** from 0.3333. Latency delta: -122.8 ms
(candidate faster), which is irrelevant given the quality regression.

## SDLC Phase A classification

`positive: false`, `stack_layer: false`, `action: no_stack_layer_non_positive`.

Reasons (from `sdlc_delivery.json`):

1. `fixture_insufficient_n:c5-control` (n=3 < 20)
2. `fixture_insufficient_n:c5-component-plan` (n=3 < 20)
3. `non_regression_fail:binder_reference_f1:0.7333333333333334->0.6333333333333333`
4. `primary_metric_null_or_worse:smoke.structural_similarity:control=0.23083333333333333 candidate=0.1725 improvement=-0.05833333333333335`
5. `fixture_insufficient_n_alone`

This is a genuine quality regression (not merely a null delta or fixture-`n`
artifact): the primary metric moves the *wrong* direction and a secondary
non-regression gate (`binder_reference_f1`) fails outright. Ship gates
correctly fail closed; no gate was weakened to reach this result.

## Next-run priorities

1. **model:** `component-plan` is now **rejected** for this recipe (a
   confirmed quality regression, not merely exhausted) — test the distinct
   size-matched `component-edge` hypothesis next; do not reselect
   `component-plan` without a new preregistered hypothesis or a materially
   different recipe (e.g. a different loss weight or step budget).
2. **evaluation:** keep the matched control as the size-matched baseline every
   cycle.
3. **infrastructure:** soft ship-gate fails on fixture `n` never stop the
   continuous loop.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c5/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c5-{control,component-plan}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-20260802-c5-results.json`
- Predecessor (incomplete): [cycle 3 results](continuous-openui-local-20260802-c3-results.md)
- Harness fix: `scripts/run_autotrain_continuous.py` commit `56fb65a7e6a458d6a470bd2104c2486144784293`
