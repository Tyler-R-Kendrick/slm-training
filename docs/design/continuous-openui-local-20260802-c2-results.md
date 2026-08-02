# Continuous autotrain cycle 2 results (2026-08-02, loop `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c2` |
| Cycle intent | `retry_measurement` — frozen replay of cycle 1's control/candidate arm |
| Upstream / integration | `b8188a49` / `2a4e7fa6` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke`, `--ship-gates` |
| Wall cap | 3 minutes |

## Why this cycle exists

Cycle 1 ([results](continuous-openui-local-20260802-c1-results.md)) trained both
arms successfully but never produced a ship-gate `scoreboard.json` for either
arm: this sandbox's ambient `NODE_OPTIONS="--import tsx"` made every `node`
subprocess exit 9, so the AgentV publish step inside
`scripts.evaluate_model --ship-gates` raised `AgentV SDK is unavailable`. That
was fixed locally (`env -u NODE_OPTIONS npm ci`, installs untracked
`node_modules/@agentv/core`; no repository code change) and this cycle re-ran
the *identical* frozen control/candidate arm with `NODE_OPTIONS` unset for the
driver subprocess, per the `retry_measurement` handoff action and the
continuous-loop frozen-replay contract.

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c2-control | bounds off | 3 | 1.0 | 0.0 | 0.0575 | 1409.95 | **fail** (insufficient_n, quality thresholds) |
| c2-bounds | bounds **on** | 3 | 1.0 | 0.0 | 0.0575 | 1368.81 | **fail** (same) |

`measurement_complete: true` — both arms now have a full AgentV ship-gate
scoreboard (`gates.pass=false`, `held_out`/`adversarial`/`ood`/`rico_held`
suites `missing_suite` at smoke scope, as expected).

Primary delta (bounds − control) `structural_similarity`: **0.0** (flat, ties
c1). Latency delta: **-41.1 ms** (candidate faster), but
`meaningful_program_rate` stays **0.0** on both arms — same pure latency blip
pattern as c1, now confirmed rather than inferred.

## SDLC Phase A classification

`positive: false`, `stack_layer: false`, `action: no_stack_layer_non_positive`.

Reasons (from `sdlc_delivery.json`):

1. `fixture_insufficient_n:c2-bounds` (n=3 < 20)
2. `fixture_insufficient_n:c2-control` (n=3 < 20)
3. `primary_metric_null_or_worse:smoke.structural_similarity:control=0.0575 candidate=0.0575 improvement=0.0`
4. `fixture_insufficient_n_alone`

Fixture `insufficient_n` and a null primary-metric delta are explicitly **not
positive** per `sdlc` autotrain-iteration-delivery. Ship gates correctly fail
closed; no gate was weakened to reach this result.

## Next-run priorities

1. **model:** `grammar_completion_bounds` is exhausted for this recipe (two
   completed measurements, one replay-confirmed) — test the distinct
   size-matched `component-plan` quality hypothesis next; do not reselect
   `bounds` without a new preregistered hypothesis.
2. **evaluation:** keep the matched control as the size-matched baseline every
   cycle.
3. **infrastructure:** soft ship-gate fails on fixture `n` never stop the
   continuous loop.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c2/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c2-{control,bounds}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-20260802-c2-results.json`
