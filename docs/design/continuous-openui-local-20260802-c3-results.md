# Continuous autotrain cycle 3 results (2026-08-02, loop `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c3` |
| Cycle intent | `screening` — first size-matched test of the `component-plan` hypothesis (c2's rank-1 next-run priority; `grammar_completion_bounds` is exhausted for this recipe) |
| Upstream / integration | `b8188a49` / `2af415dc` |
| Device | CPU |
| Steps | 20 / seed 100003 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke`, `--ship-gates` |
| Wall cap | 3 minutes |
| Hypothesis | Component-plan supervision improves smoke `structural_similarity` without lowering `parse_rate` or `binder_reference_f1` |
| Primary metric | `smoke.structural_similarity` (direction: increase, minimum_effect 0.01) |

## Run matrix (partial, single-suite smoke diagnostic only)

| Arm | Levers | Params | smoke n | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c3-control | component-plan off | 1,755,764 | 3 | 1.0 | 0.3333 | 0.2308 | 0.7333 | 3790.32 | eval **incomplete**: ship-gate `scoreboard.json` never written |
| c3-component-plan | component-plan **on** | 1,755,764 | 3 | 1.0 | 0.0 | 0.1725 | 0.6333 | 3701.36 | eval **incomplete**: ship-gate `scoreboard.json` never written |

Partial primary delta (component-plan − control) `structural_similarity`: **-0.0583** (candidate worse). `binder_reference_f1` also regresses (0.7333 → 0.6333) and `meaningful_program_rate` drops to 0.0 on the candidate. Latency delta: -89.0 ms (candidate faster), irrelevant given the quality regression. **These numbers are from the partial single-suite diagnostic only — treat as directional, not conclusive, until the full ship-gate scoreboard completes (see blocker below).**

## Blocker: local AgentV SDK / NODE_OPTIONS gap recurs (not a repo code bug)

Same root cause as cycles 1-2: this sandbox's ambient `NODE_OPTIONS="--import tsx" --max-old-space-size=8192` makes every plain `node` invocation fail (`node: --import tsx is not allowed in NODE_OPTIONS`), so the AgentV publish step inside `scripts.evaluate_model --ship-gates` raised `AgentV SDK evaluation failed: node: --import tsx is not allowed in NODE_OPTIONS` for both arms. `node_modules/@agentv/core` is already installed from the cycle-1 `npm ci` fix, so this is *not* a missing-dependency repeat — it recurred because this cycle's first supervised driver invocation (`python -m scripts.run_autotrain_continuous --supervised ...`) was run from a shell that still exported the ambient `NODE_OPTIONS`, and that env is inherited by every subprocess in the tree including the AgentV `node` publish step. Fix applied: re-ran the identical driver invocation as `env -u NODE_OPTIONS python -m scripts.run_autotrain_continuous ...` so `NODE_OPTIONS` is unset for the whole process tree, not just an `npm ci` call. This is a one-time local shell-invocation fix, not a repository code change, and no `HarnessSignalV1` was raised (per operator guidance, this recurring env quirk does not name a canonical harness family).

Residual risk: any future supervised invocation of this driver must be launched with `NODE_OPTIONS` unset or the ship-gate publish step will fail again with the identical `missing_scoreboard` symptom.

## SDLC Phase A classification

`positive: false`, `stack_layer: false`, `action: no_stack_layer_non_positive`.

Reasons (from `sdlc_delivery.json`):

1. `measurement_incomplete:c3-control:missing_scoreboard`
2. `measurement_incomplete:c3-component-plan:missing_scoreboard`
3. `non_regression_fail:binder_reference_f1:0.7333333333333334->0.6333333333333333`
4. `primary_metric_null_or_worse:smoke.structural_similarity:control=0.23083333333333333 candidate=0.1725 improvement=-0.05833333333333335`

Measurement is incomplete (no full ship-gate scoreboard for either arm), so this cycle is diagnosed as `infrastructure` / `measurement_incomplete`, not a concluded model result. The partial numbers point toward a regression, but per the frozen-replay contract they are not treated as model evidence until the full scoreboard completes.

## Next-run priorities

1. **infrastructure:** replay the identical frozen control/component-plan arm with `NODE_OPTIONS` unset for the whole driver process tree to complete the missing `scoreboard.json` measurement before drawing any model conclusion (see cycle 4 below, completed in the same session).
2. **model:** do not accept or reject the `component-plan` hypothesis on this partial diagnostic; wait for the completed replay.
3. **evaluation:** keep ship gates honest and fail-closed on fixture `n` / missing suites.

## Addendum: replay attempt (cycle 4) crashed on a harness bug, fixed and replayed as cycle 5

The first automatic `retry_measurement` attempt (would-be campaign `...-c4`) crashed
before producing any scoreboard with `RuntimeError: unsupported automatic frozen
replay arm: plan`. Root cause: `scripts.run_autotrain_continuous._apply_frozen_replay`
recovered the screening-bank arm slug from the frozen experiment id via
`rsplit("-", 1)[-1]`, which only keeps the token after the *last* hyphen — for
`...-c3-component-plan` that yields `"plan"`, which is not a bank member (half of
`_SCREENING_ARM_BANK`'s slugs are hyphenated: `component-plan`, `component-edge`,
`component-inventory`, `binder-topology`, `component-structure`). This reproduced on
the frozen input and named exactly one canonical family (`autoresearch`), so it was
routed through `improve-openui-harnesses`: fixed in `scripts/run_autotrain_continuous.py`
(commit `56fb65a7e6a458d6a470bd2104c2486144784293`) by recovering the slug from the
known source-campaign prefix (falling back to the longest matching bank slug by
suffix), with two new regression tests
(`test_frozen_replay_resolves_hyphenated_arm_slug`,
`test_screening_arm_slug_prefers_source_campaign_prefix`) and a `harness.autoresearch.experiment_campaign`
v59→v60 bump. The identical frozen arm was then replayed successfully as campaign
`continuous-loop-20260802-continuous-openui-local-8c0b60dd-c5` — see
[cycle 5 results](continuous-openui-local-20260802-c5-results.md).

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c3/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c3-{control,component-plan}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-20260802-c3-results.json`
- Followed by the frozen replay: [cycle 5 results](continuous-openui-local-20260802-c5-results.md)
