# continuous-openui-local c2 — canvas thrash null

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Cycle | 2 |
| Campaign | `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2` |
| Integration | `07274048f` (on `bf31eb7a3` + generate_batch_size fix) |
| Role | screening |
| Positive | **no** (fixture null + insufficient_n) |
| Stack layer | no |

## Recipe

- train_version `wf_smoke_v2`, eval smoke n=3, CPU scratch, steps≈22, seed 100002
- size-matched arms @ ~1.609e6 params
- screening `generate_batch_size=1` (schema now accepts it)

## Results

| Arm | structure | MPR | binder F1 | parse | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0.327 | 0.0 | 0.0 | 1.0 | 7359 |
| canvas | 0.327 | 0.0 | 0.0 | 1.0 | 7388 |

Primary `smoke.structural_similarity` delta **0.0**. Expected fixture ship-gate fails (`insufficient_n`, quality). Climb rejected; ship blocked.

## Harness

Cycle 1 failed validating `generate_batch_size` (forbidden knob). Repair landed in
`07274048f`; this cycle is the first successful dual-arm measurement after that
unblock. Experiment itself is non-positive.

## Next

Ranked successor: **component-plan** thrash vs matched control.
