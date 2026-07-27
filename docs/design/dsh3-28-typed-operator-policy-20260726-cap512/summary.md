# DSH3-28 typed dynamic operator policy (SLM-403)

Status: bounded local measured result; not a ship claim

## Matrix

| Arm | Seed | Action | Action + arguments | Singleton forwards | Partial forced |
| --- | ---: | ---: | ---: | ---: | ---: |
| `zero` | 11 | 0.500 | 0.000 | 0 | 0 |
| `random` | 11 | 0.000 | 0.000 | 0 | 0 |
| `shuffled_labels` | 11 | 0.000 | 0.000 | 0 | 0 |
| `enabled` | 11 | 0.500 | 0.000 | 0 | 0 |

## Honesty

This run uses 2 local train policy rows and 2 held-out policy rows, bounded to CPU. It is an integration/control result only: no checkpoint, human rating, remote workload, or ship-gate claim. CAP2 v1 replay drift remains explicitly outside this current-surface matrix.

Decision: `measured` — bounded local control matrix completed; this is not a ship decision
