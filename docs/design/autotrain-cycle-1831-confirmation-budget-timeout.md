# Autotrain c1831: confirmation budget timeout

**Verdict:** no model result. The exact c1830 fresh-seed confirmation stopped
before either arm executed because the driver reserved the full 70-second
screening allowance for each arm *after* Git sync, evidence capture, research,
and hypothesis compilation had already spent part of the 180-second cycle cap.

| Stage | Status | Signal |
| --- | --- | --- |
| Latest / merge | pass | upstream `b0587635...78a83` integrated as `2fe1826e...e957` |
| Frozen confirmation plan | pass | exact capacity-aware control and tail-supervision candidate, new seed |
| Research / hypotheses | pass | five-candidate matrix formed; confirmation selected |
| Symmetric arm reservation | timeout | requested `70s + 70s + 15s finalization = 155s`; less than 155s remained |
| Train / eval / AgentV | not run | no arm started; no metric, checkpoint, or learning claim |
| Lean / formal | `not_applicable:pre_execution` | promotion preflight was not reached |

This is a harness failure, not a negative model result. The guard correctly
refused an asymmetric run, but its static reservation made the legal
confirmation needlessly infeasible: it computed the arm allowance before the
planning stages and required that full allowance again afterward. Campaign
harness v128 now fits both arm budgets equally to the actual post-planning time
remaining, caps each subprocess to that fitted share, and retains the canonical
15-second finalization reserve. It does not extend or weaken the three-minute
repository cap.

Next: rerun the exact c1830 confirmation on a new seed under v128. Only a
complete two-arm run may confirm or reject the candidate. Machine evidence:
[`autotrain-cycle-1831-confirmation-budget-timeout.json`](autotrain-cycle-1831-confirmation-budget-timeout.json).
