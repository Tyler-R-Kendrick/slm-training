# Autotrain c1784: component-plan confirmation rejects the candidate

**Verdict:** reject and exhaust the component-plan family. On fresh seed
101784, the candidate and size-matched control tie exactly on smoke and
held-out quality. The candidate is 9.19% faster on smoke, where meaningful
program rate is zero, and only 2.92% faster on held-out, below the 5%
efficiency screen. Candidate training takes 2.38x as long.

| Arm | Params / train | Smoke | Held-out | Decision |
| --- | --- | --- | --- | --- |
| component-plan confirm | 1,755,764; 21 steps; loss 15.50356; 6.52 s | n=3; parse 1; meaning 0; structure .05750; binder .8222; p50 1,246.97 ms | n=5; parse 1; meaning .2; structure .08940; binder .7076; p50 1,249.79 ms | reject |
| matched control | 1,755,764; 21 steps; loss 11.33302; 2.74 s | identical quality; p50 1,373.19 ms | identical quality; p50 1,287.42 ms | baseline |

Both AgentV bundles are complete with zero execution errors and both ship-gate
reports fail. Treatment integrity also holds: training and evaluation agree on
compact canvas off, grammar completion bounds off, and component-plan decode
weight 1 versus 0. The different absolute metrics from c1783 are seed variance,
not a harness mismatch.

This CPU fixture rejects c1783's efficiency candidate: its held-out speed gain
does not clear the preregistered minimum and no quality gain appears. Lower
training loss remains diagnostic only; it is not promotion evidence. Both
explicit no-sync scratch checkpoints are provenance-only and must not be
reused, promoted, synced, or shipped. Lean is
`not_applicable:confirmation`; no champion exists for formal preflight.

The next substantive experiment should preregister a distinct size-matched
objective that directly targets certified structural or meaningful-program
quality. A non-exhausted batch-size arm may run only as a runtime diagnostic,
not as a quality hypothesis.

Machine evidence:
[`autotrain-cycle-1784-component-plan-confirmation-rejection.json`](autotrain-cycle-1784-component-plan-confirmation-rejection.json).
