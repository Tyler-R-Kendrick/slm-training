# Autotrain c4: evaluation wall-timeout, non-positive

**Verdict:** non-positive infrastructure timeout. Both `control` and `steps`
arms of cycle 4 (loop `continuous-openui-20260802-local`) completed
training and wrote checkpoints, but evaluation
(`--suites smoke,held_out --ship-gates`, real AgentV decode) exceeded the
repo-wide `MAX_RUN_MINUTES=3` wall cap before producing a scoreboard for
either arm.

## Result matrix

| Arm | Train | Eval | Disposition |
| --- | --- | --- | --- |
| control | completed (checkpoint written) | wall-timeout, empty metrics | Not scoreable |
| steps | completed (checkpoint written) | wall-timeout, empty metrics | Not scoreable |

## Why this is not a stack-layer positive despite the driver's headline

The driver's `cycle_handoff.json` lists both
`primary_metric_win:held_out.structural_similarity:0.4166->0.51` and
`efficiency_win:mpr_per_ms` among its reasons, and the initial log line
reads `SDLC_PHASE_A POSITIVE`. Neither of this cycle's own outcome files
(`control`, `steps`) has any metrics — both are `status=stopped,
metrics={}` — so that headline does not trace to a scoreboard this cycle
actually produced. The driver's own follow-up action already reflects that:
`action=positive_no_tracked_delta_skip_stack`. Per
`autotrain-iteration-delivery.md`'s "when uncertain, treat as not positive"
rule and the explicit non-positive case "wall timeouts with no metric win,"
this cycle is documented as **non-positive** and does not open a new stack
layer.

## Next step

Retry the identical frozen manifest
(`c55de2b4b135f16791fa5a9b6127456ba3e18a522187117571fcf7fdde66c9ae`) with a
narrower eval scope (`smoke` only) or a longer `--decode-timeout-seconds` so
evaluation fits inside the wall cap; two full suites with honest ship gates
and real AgentV publication does not currently fit `MAX_RUN_MINUTES=3` on
this CPU-only container.

Eval commit: `df78abae7c803d84b6fad39bcf9179f447203ab0`.
