# Autotrain 1pse54 c2: AgentV NODE_OPTIONS harness block (infrastructure, repaired)

**Evidence class:** local CPU fixture/scratch. **Disposition:** infrastructure
failure; no model-quality, promotion, or ship claim. Machine-readable evidence:
[`autotrain-cycle-1pse54-c2-agentv-node-options-block.json`](autotrain-cycle-1pse54-c2-agentv-node-options-block.json).

## Result matrix

Both the control and `grammar_completion_bounds` candidate arm of
`continuous-loop-20260802-continuous-openui-1pse54-c1f3ee08-c2` trained to
their declared 21 steps on `wf_smoke_v2`, seed 100001, batch size 2, 1,608,962
matched parameters, and wrote local no-sync checkpoints. Evaluation never
started on either arm.

| Arm | Params | Last loss | Train status | Eval / AgentV | Gates | Decision |
| --- | ---: | ---: | --- | --- | --- | --- |
| control | 1,608,962 | 22.6219 | complete | blocked | unavailable | infrastructure / inconclusive |
| bounds (`grammar_completion_bounds`) | 1,608,962 | 22.6219 | complete | blocked | unavailable | infrastructure / inconclusive |

Loss is training telemetry only; both arms tie exactly and are not evidence of
a quality difference since neither ran evaluation.

## Root cause and fix

`scripts.evaluate_model --ship-gates` failed identically on both arms with
`AgentV SDK evaluation failed: node: --import tsx is not allowed in
NODE_OPTIONS`, raised from `publish_agentv_evaluation` in
`src/slm_training/evals/agentv.py`, before any AgentEvals criterion ran. The
session container's inherited `NODE_OPTIONS` is malformed for this Node
build; `src/slm_training/dsl/grammar/backends/graphql_js.py` already carries
a `_sanitized_env()` fix for the identical class of bug in the graphql-js
bridge, but `agentv.py`'s subprocess call never adopted it.

Fix (commit `46fecfc`): added the same `_sanitized_env()` helper to
`agentv.py` and passed it to the AgentV `subprocess.run` call
(`evals.agentv` v6 → v7), plus a regression test asserting the child env
carries `NODE_OPTIONS=""`. While in the file, also repaired two
`test_agentv.py` fixtures left stale by the newer `ast_beq_rate` /
`canonical_beq_rate` ship-gate criteria — their `smoke` suite dicts omitted
those keys, which synthesized extra missing-metric failures unrelated to what
each test actually checks, and had blocked at least two prior sessions'
identical NODE_OPTIONS fix from landing (see PR #1316, #1282) because the
repo's pre-commit hook runs every test in a changed file.

## Next step

Replay the identical frozen `c2` control/bounds arms with the repaired
harness before evaluating any new hypothesis — no knob changes.

Both checkpoints referenced here (`runs/c20260802-continuous-openui-1pse54-c1f3ee08-c2-{control,bounds}/checkpoints/last.pt`)
are local, explicit no-sync diagnostics. Neither is reusable, promoted, or
ship evidence. Lean is `not_applicable:no_champion`.
