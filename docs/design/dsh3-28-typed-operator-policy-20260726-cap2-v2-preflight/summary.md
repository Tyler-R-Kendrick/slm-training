# DSH3-28 typed dynamic operator policy (SLM-403)

Status: bounded local measured result; not a ship claim

## Matrix

| Head | Arm | Seed | Action | Action + arguments | Singleton forwards | Partial forced |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `local_flat` | `zero` | 11 | 0.000 | 0.000 | 0 | 0 |
| `local_flat` | `random` | 11 | 0.000 | 0.000 | 0 | 0 |
| `local_flat` | `shuffled_labels` | 11 | - | - | - | - |
| `local_flat` | `enabled` | 11 | - | - | - | - |

## Honesty

This run uses 4 local train policy rows and 4 held-out policy rows, bounded to CPU. It is an integration/control result only: no checkpoint, human rating, remote workload, or ship-gate claim. CAP2 v1 replay drift remains explicitly outside this current-surface matrix.

Decision: `reject` — no COMPLETE local training rows; enabled and shuffled-label arms were not run
