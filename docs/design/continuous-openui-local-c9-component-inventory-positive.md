# continuous-openui-local c9 — component-inventory **positive** (fixture)

| Field | Value |
| --- | --- |
| Cycle | 9 |
| Positive | **yes** (primary + efficiency; quality held) |
| Stack layer | no (no new tracked code delta this cycle) |
| Params | ~1.682e6 size-matched |

## Results (smoke n=3)

| Arm | structure | MPR | binder F1 | parse | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0.2769 | 0.667 | 1.000 | 1.0 | 6404.4 |
| component-inventory | 0.4480 | 0.667 | 1.000 | 1.0 | 4874.3 |

- Primary Δ structure **+0.1711**
- Efficiency win mpr/ms **+31.4%** with quality held (parse 1.0, MPR 0.667)
- Champion queued: `champ-continuous-openui-local-9-96cb85cae43e4d25`

## Honesty

Fixture-scale only. **Fresh-seed confirmation required** before any promote claim. Never ship from this n=3 board alone.

## Next

Confirm component-inventory on a fresh seed with exact size-matched recipes.
