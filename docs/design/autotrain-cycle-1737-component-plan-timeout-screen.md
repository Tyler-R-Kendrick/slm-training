# Autotrain c1737: component-plan screen (measurement incomplete)

**Verdict:** no valid comparison. The matched control's smoke-suite decode
did not finish scoring within this session's `MAX_RUN_MINUTES` cap (2 of 3
records processed before the run interrupted), so `primary_metric` has no
control value to compare against. The `component_plan_loss_weight=1.0`
candidate did complete and honestly fails ship gates on its own — parse 1.0
but `binder_reference_f1=0`, `meaningful_program_rate=0`, and every quality
gate below floor on fixture `n=3`. This is a soft, non-terminal wall-timeout
per the continuous-loop rules, not evidence the lever is bad; it is simply
unmeasured against a control this cycle.

## Result matrix

| Arm | n | Parse | Binder F1 | Meaningful | Structure | p50 | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 2/3 processed | — | — | — | — | — | decode interrupted by wall cap; not scoreable |
| `component_plan_loss_weight=1.0` | 3 | 1.0 | 0 | 0 | .0964 | 2,878.71 ms | complete; fails every quality gate |

Training completed for both arms (control loss 14.3902 / 3.87s wall; treatment
loss 19.9160 / 9.32s wall — CPU scratch, 22 steps, seed 100001, batch 2). The
gap is entirely in evaluation: compiler-tree decode on this session's CPU is
slow enough that even the 3-record smoke suite does not always finish inside
the per-command cap.

`--ship-gates` fail on the completed arm (fixture `n=3` well below the
evidence floor; `held_out`/`adversarial`/`ood`/`rico_held` were not run). Lean
is `not_applicable:screening`; promotion and RL remain locked.

## Harness signal

No repair applied — a wall-timeout on one arm is an expected soft failure
under the continuous-loop rules (`docs/design/` "Soft failures ... never stop
the loop"), and the driver correctly rotated to the next screening arm rather
than re-running the same interrupted measurement. Recorded here as a data
point: CPU compiler-tree smoke decode is close enough to the cap that
individual arms can miss it non-deterministically, which the harness may want
to account for (e.g. a slightly larger per-eval budget or a smaller
`decode-timeout-seconds`) if this recurs across future cycles.

## Repaired next priority

Per the cross-cycle screening-arm bank, the next supervised cycle should test
the standalone `component-edge` hypothesis:

> Area `model`: `component_edge_loss_weight=1.0` improves smoke
> `structural_similarity` without lowering `parse_rate` or
> `binder_reference_f1`, versus the matched control.

Machine-readable evidence is in
[`autotrain-cycle-1737-component-plan-timeout-screen.json`](autotrain-cycle-1737-component-plan-timeout-screen.json).
