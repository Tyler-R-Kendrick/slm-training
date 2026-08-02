# Autotrain 1pse54 c4: promotion-role decode timeout (infrastructure, not a model result)

**Evidence class:** local CPU fixture/scratch. **Disposition:** infrastructure
signal; no model-quality, promotion, or ship claim. Machine-readable evidence:
[`autotrain-cycle-1pse54-c4-promotion-decode-timeout.json`](autotrain-cycle-1pse54-c4-promotion-decode-timeout.json).

## Result matrix

Both the `control` and `component-plan` arms of
`continuous-loop-20260802-continuous-openui-1pse54-c1f3ee08-c4` (promotion
role, 1,755,764 matched params, 21 steps) trained to completion, but every
document on both arms — 3/3 smoke and 5/5 held_out — hit the per-document
decode timeout during evaluation. `completed_document_n=0` on all four suite
runs.

| Arm | Params | Last loss | Smoke complete/timeout | Held-out complete/timeout | Decision |
| --- | ---: | ---: | --- | --- | --- |
| control | 1,755,764 | 8.8520 | 0/3 | 0/5 | infrastructure / inconclusive |
| component-plan | 1,755,764 | 12.5051 | 0/3 | 0/5 | infrastructure / inconclusive |

Loss is training telemetry only and is not comparable as a quality signal
since neither arm produced a single completed evaluation document.

## Diagnosis

This is the AgentV NODE_OPTIONS fix's evaluation bridge working correctly
(0 execution errors, every record reached a finalized disposition) — the
timeout is inside the model_build decode stage itself, not the bridge fixed
in commit `46fecfc`. This is the first occurrence of this exact signal in
loop `continuous-openui-1pse54`; per the repeated-blocker rule (three
identical failures with no new information) it is **not yet a hard block**.
The `promotion` cycle role runs a larger `held_out` suite than the
`screening`-role cycles 1-3, which completed cleanly under the same decode
timeout budget, so container CPU throughput under the promotion recipe is the
leading suspect rather than a code regression.

## Next step

Replay the identical frozen `c4` control/`component-plan` arms after
confirming or repairing runtime capacity. Do not treat the 0/8 completed
documents as model evidence in either direction.

Both checkpoints are local, explicit no-sync diagnostics. Neither is
reusable, promoted, or ship evidence. Lean is `not_applicable:no_champion`.
