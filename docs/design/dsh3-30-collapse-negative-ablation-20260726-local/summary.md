# DSH3-30 collapse negative ablation (SLM-405)

Status: bounded local measured result; not a ship claim

## Matrix

| Head | Arm | Seed | Action | Action + arguments | Singleton forwards | Partial forced |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| - | - | - | - | - | - | - |

No SLM-405 training arm ran: the frozen replay-negative preflight failed closed.

## Honesty

This run uses 2 local train policy rows and 2 held-out policy rows, bounded to CPU. It is an integration/control result only: no checkpoint, human rating, remote workload, or ship-gate claim. CAP2 v1 replay drift remains explicitly outside this current-surface matrix.

Decision: `reject` — missing replay-verified matched negative strata: train:different_result, dev:different_result
