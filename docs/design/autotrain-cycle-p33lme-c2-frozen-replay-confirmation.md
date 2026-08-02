# Autotrain continuous-openui-p33lme c2: frozen replay confirms the AgentV repair

**Outcome:** non-positive model measurement (fixture `insufficient_n`,
`smoke.structural_similarity` tie); **the harness repair is replay-confirmed**
— the identical control/candidate arms that hit the AgentV `NODE_OPTIONS`
blocker in cycle 1 now complete AgentV publish and produce a full scoreboard.

## What happened

Cycle 2 is the driver's mandatory `retry_measurement` consumption of cycle
1's incomplete arms (`FROZEN_REPLAY_ACK
source_campaign=continuous-loop-20260802-continuous-openui-p33lme-489d3aa7-c1
successor_campaign=continuous-loop-20260802-continuous-openui-p33lme-489d3aa7-c2`),
run against integration commit `3861725` (docs) after the harness fix
`a82eea405548cea5fcc486c3332823e2e3b008cf` landed. It is **not** a new model
hypothesis — `next_experiment` for the distinct `component-plan` hypothesis is
queued as this cycle's rank-1 priority, not yet executed.

Both arms trained (CPU, `wf_smoke_v2`, matched size) and evaluated `smoke`
(`n=3`, strict compiler-tree policy) to a **complete** AgentV bundle
(`execution_errors=0`), then an honest `--ship-gates` rejection on fixture
`n` and quality thresholds — exactly the expected outcome for a 3-document
smoke fixture, not a harness failure:

| Arm | latency p50 (ms) | parse_rate | meaningful_program_rate | structural_similarity | ast_beq_rate | canonical_beq_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control (`...-c2-control`) | 1366.17 | 1.0 | 0.0 | 0.0575 | 0.0 | 0.0 |
| candidate (`...-c2-bounds`) | 1311.71 | 1.0 | 0.0 | 0.0575 | 0.0 | 0.0 |

Gate failures: `smoke:insufficient_n actual=3 need>=20`,
`smoke:meaningful_program_rate`, `smoke:structural_similarity`,
`smoke:component_type_recall`, `smoke:ast_beq_rate`,
`smoke:canonical_beq_rate`, `smoke:reward_score` all below threshold, plus
`held_out`/`adversarial`/`ood`/`rico_held` missing (screening cycle never ran
those suites). `expected_gate_rejection=true` — this is honest fixture-scale
behavior, not a claim of readiness.

## Why this confirms the repair

Cycle 1's identical arms raised `RuntimeError: AgentV SDK evaluation failed:
node: --import tsx is not allowed in NODE_OPTIONS` before any scoreboard was
written (`measurement_incomplete` on both arms). After
`a82eea405548cea5fcc486c3332823e2e3b008cf` (`evals.agentv` v6→v7:
`_sanitized_env()` clears `NODE_OPTIONS` before spawning the AgentV node
runner), this replay's `scoreboard.json`, `gates.json`, and AgentV
`benchmark.json` all wrote successfully for both arms —
`measurement_complete: true` in `sdlc_delivery.json`. This satisfies the
"executable unblocking" positive-result criterion from
[`autotrain-iteration-delivery.md`](../../.claude/skills/sdlc/references/autotrain-iteration-delivery.md)
for the **harness fix itself**: "a harness/path/code fix removes a prior hard
path error or unrecoverable blocker and the identical arm then completes with
a usable scoreboard (replay-proven)."

The driver's own automated `_classify_metric_tradeoff` /
`classify_positive_metrics` decision for this campaign is `positive=false`
(`stack_action=no_stack_layer_non_positive`) — its `executable_unblock`
signal only fires for an asymmetric same-campaign control-errored /
candidate-succeeded split, not for a cross-cycle "was broken, now fixed"
comparison, so it correctly reports the **model** comparison as non-positive
(fixture `n`, null primary-metric delta). No new stack layer is opened for
model results here; per this iteration's delivery, the fix itself already
shipped as commit `a82eea4` (see
[the c1 repair doc](autotrain-cycle-p33lme-c1-agentv-node-options-repair.md)).

## Next hypotheses

Per this cycle's rank-1 `NextRunPriorityV1`: run the size-matched
`component-plan` quality hypothesis
(`c20260802-continuous-openui-p33lme-489d3aa7-c2-component-plan`) as the next
model-hypothesis cycle (cycle 3), keeping the matched control as baseline.
Not executed in this delivery — reserved for the next scheduled iteration.

Machine-readable evidence is in
[`autotrain-cycle-p33lme-c2-frozen-replay-confirmation.json`](autotrain-cycle-p33lme-c2-frozen-replay-confirmation.json).
