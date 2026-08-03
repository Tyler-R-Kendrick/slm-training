> **Correction (see [c4](continuous-openui-local-n8vwtq-c4-results.md)):** the
> "control=off vs component-plan=tree" diagnosis below is wrong — both arms
> use `compiler_decode_mode="tree"`; ship-gate eval always runs under `"tree"`
> regardless of any training-time knob. The fix in commit `071560ee` is a
> reasonable general safety net but does not resolve this specific blocker
> (which recurred a 3rd time in c4 after this repair).

# Continuous autotrain: 2026-08-03 cycle 3, session n8vwtq (harness repair, exhaustion)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c3`
**Base commit:** `e82964bf` (cycle-2 docs commit)
**Intent:** `retry_measurement` — eval-only replay of the frozen `c2`
component-plan-vs-control arm (checkpoints reused, no retraining).

| Arm | Eval status | structural_similarity |
| --- | --- | --- |
| control | completed 3/3 documents | 0.32667 (matches every prior session's control) |
| component-plan | timed out mid-decode, no scoreboard | — |

**Second consecutive timeout on the identical frozen arm.** This exhausted
`measurement.max_consecutive_frozen_replays` and the driver correctly emitted
a typed `repair_harness` action (`harness_family=model_build`) instead of
retrying forever or scoring the timeout as a negative model result.

## Root cause and fix

Commit `071560ee9da5918f6b15ff71b7c6b4a66bb7265f`
([`scripts/run_autotrain_continuous.py`](../../scripts/run_autotrain_continuous.py)):
the per-arm eval-stage wall budget (`_fit_symmetric_arm_budget`) split the
fixed cycle wall budget **evenly** across arms regardless of
`compiler_decode_mode`. Tree-mode constrained decode measures ~35.7s/document
on this host's CPU vs a few seconds/document for off-mode
(`docs/design/continuous-openui-local-n8vwtq-c2-results.md`), so the even
split starved the tree-mode `component-plan` arm while its off-mode `control`
finished with time to spare.

Added `_fit_decode_weighted_arm_budgets`, which reallocates the **same total
wall-time pool** by each arm's decode-mode cost (tree gets 4x the off-mode
share) instead of splitting evenly, and wired the per-arm result into both
the child `--experiment-wall-seconds` argument and the outer
`_arm_execution_deadline` stage-kill deadline — the latter still used the
old uniform value, which would have silently capped the fix at the original
budget. `MAX_RUN_MINUTES` and the overall cycle deadline are unchanged; only
the split across arms.

Regression tests:
`tests/test_scripts/test_run_autotrain_continuous.py::test_decode_weighted_arm_budget_gives_tree_mode_more_wall_time`
and `::test_decode_weighted_arm_budget_is_symmetric_when_modes_match`.

Per `sdlc` autotrain-iteration-delivery: this is a harness repair, not a
positive model result — no stacked PR for this cycle's model claim; the
repair itself lands as an ordinary commit (already pushed alongside these
docs).

## Next priorities

1. `retry_measurement` — replay the identical frozen `component-plan` arm now
   that the eval budget scales by decode mode.
2. `component-plan` is already independently reproduced positive 3 times on
   other hosts (#1369, #1376, #1378); this replay is corroboration, not the
   original discovery.

Machine evidence:
[`continuous-openui-local-n8vwtq-c3-results.json`](continuous-openui-local-n8vwtq-c3-results.json).
