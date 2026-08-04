# Autotrain c1768: frozen component-plan replay confirms runtime defect

**Verdict:** infrastructure-inconclusive; the identical replay budget is
exhausted and the canonical model-build runtime must be repaired before one
validation replay. Cycle c1768 reused the exact c1766 candidate and control
checkpoints and replayed their content-bound manifests. No training ran and no
new checkpoint was created.

| Arm | Frozen checkpoint | Smoke | Decode work | AgentV / gates | Decision |
| --- | --- | --- | --- | --- | --- |
| component-plan | `8af2c84d...c2e3` | n=3; completed 0; timeouts 3; p50 including incomplete 11,179.11 ms; quality unavailable | 301 forwards; 110,062 states; 110,888 transition misses; 6,289 witness expansions; 115,104 parser forks | bundle complete; no execution errors; quality gate result is not scoreable | infrastructure-incomplete |
| matched control | `35869ea8...edd8` | n=3; completed 3; parse 1; meaning .3333; structure .13527; binder F1 .6333; p50 1,214.57 ms | 11 forwards; 31,115 states; 31,277 transition misses; 861 witness expansions; 32,123 parser forks | bundle complete; no execution errors; fail | complete gate rejection |

This replay removes training variance from the diagnosis. The candidate uses
the same `8af2c84d...c2e3` checkpoint as c1766 and again times out on all three
documents; the control uses the same `35869ea8...edd8` checkpoint and again
completes all three. Candidate quality remains null by the strict completeness
contract, so this is not evidence that component-plan quality is better or
worse.

The trace identifies a causal runtime shape. The control selects a `Button`
root and closes it. The component-plan bias changes the root to a legal `Stack`
and repeatedly selects additional legal children instead of array closure. Its
three plan-induced choice changes amplify work to 27.4x the forwards, 3.54x the
unique completion states, 7.30x the witness expansions, and 3.58x the parser
forks of control. This is not primarily a cache miss: only 55 state-intern hits
were available against roughly 110,000 unique states, so the selected decode
trajectory itself expands.

The resulting harness signal is that an unbounded auxiliary Poisson-count
logit can dominate the base model after a short train. The canonical repair
bounds component-plan root and count biases smoothly by their configured
decode weight while leaving the exact legal domain, singleton bypass, terminal
witness authority, and finalize validation unchanged. The next replay must run
on that repaired implementation; production timeouts must not be widened.

Cycle c1767 was initialized but reached no train or eval stage because the
original replay parser split the hyphenated arm name incorrectly. The driver
recovered that initialized-only campaign and c1768 used the corrected
longest-suffix resolver. There is no c1767 metric claim.

These are local fixture diagnostics, not production evaluation or ship
evidence. The reused checkpoints remain explicit no-sync artifacts and are not
reusable, promoted, or ship candidates. Lean is
`not_applicable:no_champion`; a promotion still requires the formal preflight
and theorem-backed optimum bands.

Machine evidence:
[`autotrain-cycle-1768-component-plan-replay-incomplete.json`](autotrain-cycle-1768-component-plan-replay-incomplete.json).
