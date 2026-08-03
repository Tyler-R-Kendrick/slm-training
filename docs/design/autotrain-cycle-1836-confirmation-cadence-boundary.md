# Autotrain c1836: confirmation cadence boundary

**Verdict:** no model result. The first c1836 attempt reclaimed c1830's
confirmation and selected the correct smoke screening role, but stopped before
research because cycle 1836 was a reserved promotion slot. After the cadence
repair, the same campaign passed research and matrix locking and started both
matched arms; both exhausted their symmetric 66.55-second envelopes during
training, before any scoreboard or AgentV bundle existed.

| Stage | Status | Signal |
| --- | --- | --- |
| Latest integration | pass | origin/main ancestor of integration commit |
| Champion recovery | pass | pre-execution attempt reclaimed, attempt 2/2 opened |
| Confirmation role | pass | screening selected; promotion suites stayed closed |
| Cadence validation | repaired / pass | explicit pending-confirmation deferral only |
| Candidate train | incomplete | 16/20 steps; final recorded loss 22.566 |
| Control train | incomplete | 2/20 steps; final recorded loss 141.106 |
| Eval / AgentV | not run | neither arm reached evaluation |
| Lean / formal | `not_applicable:screening` | promotion was not authorized |

Campaign v133 adds an explicit `confirmation_pending` cadence input. It permits
only a fresh-seed confirmation to use screening suites in a reserved promotion
slot. An ordinary screening arm with an available promotion target remains a
cadence violation. Promotion endpoints, claim class, and formal requirements
are unchanged.

The run traces show a shared-host scheduling problem rather than a candidate
quality result. The host exposed 12 CPUs and a supporting sample observed
12--14 runnable workers and up to 95% user CPU; the scratch runtime configured
11 Torch threads per child. Campaign v134 pins only scratch-context train and
eval subprocesses to one OpenMP/MKL thread, identically for control and
candidate. Full-context and GPU execution are unaffected. No checkpoint was
written, and partial losses are diagnostic only.

Next: replay the exact frozen comparison under the bounded single-thread
scratch scheduler. Machine evidence:
[`autotrain-cycle-1836-confirmation-cadence-boundary.json`](autotrain-cycle-1836-confirmation-cadence-boundary.json).
