# Autotrain c1744: promotion-role eval hits the repository wall cap

**Verdict:** the loop escalated to a `promotion` cycle (`smoke,held_out`
suites) for the `component-edge` candidate vs. its size-matched control.
Both arms' `evaluate_model --ship-gates` stage exceeded the repository
`MAX_RUN_MINUTES` cap and were interrupted before a scoreboard could be
written. This is a soft timeout, not model evidence, and does not stop the
loop.

## Result matrix

| Arm | Params | Status | Partial signal | Verdict |
| --- | ---: | --- | --- | --- |
| control | 1,766,990 | wall_timeout | `held_out.structural_similarity`=0.4167 (partial) | incomplete |
| component-edge | 1,766,990 | wall_timeout | `held_out.structural_similarity`=0.4167 (partial) | incomplete |

The partial decode progress also shows an `efficiency_win` signal
(`mpr_per_ms` +9.09%, above the 5% minimum) with `quality_held`
(`parse=1.0`, `mpr=0.3333`) on the documents that did complete before the
interrupt — but per the harness's own classification this is **not**
promotable: `measurement_incomplete` on both arms and a null primary-metric
delta (`held_out.structural_similarity` 0.4167 → 0.4167) dominate. It is
recorded here only as a next-run signal, not as a positive result.

## Signals and next run

- Root cause is budget, not code: the `smoke,held_out` promotion suite is
  larger than the `smoke`-only screening suite used in c1741–c1743 and does
  not fit the same wall share under this container's timing.
- Next action is `retry_measurement`: replay the identical frozen
  `component-edge`/control arms (`cde1ec25…af292`) before drawing any
  promotion conclusion.
- If the timeout reproduces on replay, escalate to a `repair_harness` signal
  for the promotion-suite wall-budget allocation (same class of defect as
  `autotrain-cycle-1723-supervisor-budget.md`), not a model verdict.

No checkpoint was promoted; `docs/MODEL_CARD.md` / README are unchanged.
Machine-readable evidence is in
[`autotrain-cycle-1744-promotion-wall-timeout.json`](autotrain-cycle-1744-promotion-wall-timeout.json).
