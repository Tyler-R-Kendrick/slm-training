# continuous-openui-local c3 — component-plan thrash worse

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Cycle | 3 |
| Campaign | `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c3` |
| Positive | **no** |
| Stack layer | no |

## Results (smoke n=3, size-matched ~1.756e6 params)

| Arm | structure | MPR | binder F1 | parse | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0.2308 | 0.3333333333333333 | 0.733 | 1.0 | 2448.8 |
| component-plan | 0.1725 | 0.0 | 0.633 | 1.0 | 2323.1 |

Primary delta **-0.0583** (worse). Binder non-regression fail (0.733→0.633). Fixture ship gates fail as expected.

## Next

Ranked successor: **component-edge** thrash vs matched control.
