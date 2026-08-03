# Autotrain c1837: confirmation replay dispatch

**Verdict:** no model result. c1837 selected c1836's exact frozen retry and
captured current-main evidence, then stopped before hypothesis locking because
automatic replay dispatch did not recognize the champion queue's typed
`-confirm` candidate identity.

| Stage | Status | Signal |
| --- | --- | --- |
| Latest integration | pass | origin/main ancestor of integration commit |
| Frozen retry selection | pass | c1836 control/candidate manifest pair selected |
| Evidence / research | pass | current-main successor snapshot captured |
| Confirmation replay dispatch | fail | `…c1836-confirm` was not a registered replay kind |
| Train / eval / AgentV | not run | no model or metric evidence |
| Lean / formal | `not_applicable:pre_execution` | screening replay did not execute |

Campaign v135 recognizes only the typed `-confirm` replay identity in addition
to registered screening slugs and `-promote`. It derives the source lever family
from frozen typed knobs, rewrites the corresponding matrix member, and then
restores the exact frozen recipe and manifest lineage. Unknown arm identities
still fail closed.

Next: replay c1836's exact comparison under the v134 single-thread scratch
scheduler. Machine evidence:
[`autotrain-cycle-1837-confirmation-replay-dispatch.json`](autotrain-cycle-1837-confirmation-replay-dispatch.json).
