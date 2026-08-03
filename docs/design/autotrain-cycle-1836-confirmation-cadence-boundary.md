# Autotrain c1836: confirmation cadence boundary

**Verdict:** no model result. c1836 reclaimed c1830's confirmation and selected
the correct smoke screening role, but stopped before research because cycle
1836 was a reserved promotion slot and the cadence validator did not recognize
pending confirmation as a safe deferral.

| Stage | Status | Signal |
| --- | --- | --- |
| Latest integration | pass | origin/main ancestor of integration commit |
| Champion recovery | pass | pre-execution attempt reclaimed, attempt 2/2 opened |
| Confirmation role | pass | screening selected; promotion suites stayed closed |
| Cadence validation | fail | reserved promotion slot rejected screening claim |
| Train / eval / AgentV | not run | no model or metric evidence |
| Lean / formal | `not_applicable:pre_execution` | promotion was not authorized |

Campaign v133 adds an explicit `confirmation_pending` cadence input. It permits
only a fresh-seed confirmation to use screening suites in a reserved promotion
slot. An ordinary screening arm with an available promotion target remains a
cadence violation. Promotion endpoints, claim class, and formal requirements
are unchanged.

Next: retry the same new-seed smoke confirmation. Machine evidence:
[`autotrain-cycle-1836-confirmation-cadence-boundary.json`](autotrain-cycle-1836-confirmation-cadence-boundary.json).
