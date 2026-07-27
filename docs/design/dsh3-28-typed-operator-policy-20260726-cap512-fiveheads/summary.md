# DSH3-28 typed dynamic operator policy (SLM-403)

Status: bounded local measured result; not a ship claim

## Matrix

| Head | Arm | Seed | Action | Action + arguments | Singleton forwards | Partial forced |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `local_flat` | `zero` | 11 | 0.500 | 0.000 | 0 | 0 |
| `local_flat` | `random` | 11 | 0.000 | 0.000 | 0 | 0 |
| `local_flat` | `shuffled_labels` | 11 | 0.000 | 0.000 | 0 | 0 |
| `local_flat` | `enabled` | 11 | 0.500 | 0.000 | 0 | 0 |
| `ternary_ecoc` | `zero` | 11 | 0.500 | 0.000 | 0 | 0 |
| `ternary_ecoc` | `random` | 11 | 0.000 | 0.000 | 0 | 0 |
| `ternary_ecoc` | `shuffled_labels` | 11 | 0.000 | 0.000 | 0 | 0 |
| `ternary_ecoc` | `enabled` | 11 | 0.500 | 0.000 | 0 | 0 |
| `factorized` | `zero` | 11 | 0.500 | 0.000 | 0 | 0 |
| `factorized` | `random` | 11 | 0.000 | 0.000 | 0 | 0 |
| `factorized` | `shuffled_labels` | 11 | 0.000 | 0.000 | 0 | 0 |
| `factorized` | `enabled` | 11 | 0.500 | 0.000 | 0 | 0 |
| `independent_set` | `zero` | 11 | 0.500 | 0.000 | 0 | 0 |
| `independent_set` | `random` | 11 | 0.000 | 0.000 | 0 | 0 |
| `independent_set` | `shuffled_labels` | 11 | 0.000 | 0.000 | 0 | 0 |
| `independent_set` | `enabled` | 11 | 0.500 | 0.000 | 0 | 0 |
| `recurrent_set` | `zero` | 11 | 0.500 | 0.000 | 0 | 0 |
| `recurrent_set` | `random` | 11 | 0.000 | 0.000 | 0 | 0 |
| `recurrent_set` | `shuffled_labels` | 11 | 0.000 | 0.000 | 0 | 0 |
| `recurrent_set` | `enabled` | 11 | 0.500 | 0.000 | 0 | 0 |

## Honesty

This run uses 2 local train policy rows and 2 held-out policy rows, bounded to CPU. It is an integration/control result only: no checkpoint, human rating, remote workload, or ship-gate claim. CAP2 v1 replay drift remains explicitly outside this current-surface matrix.

Decision: `reject` — no typed head caused a beneficial enabled-versus-zero held-out choice change
