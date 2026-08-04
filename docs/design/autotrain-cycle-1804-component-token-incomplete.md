# Autotrain c1804: component-token screen is incomplete

**Verdict:** no attribution. Both arms trained for 21 CPU scratch steps at
1,608,962 parameters, but the matched control timed out on all 3 smoke records.
The exact frozen pair must be replayed once.

| Arm | Loss | Complete | Structure | Binder F1 | Recall | MPR | Fidelity | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 0 control | 14.95899 | 0/3 | — | — | — | — | — | — |
| component weight 1 | 16.65722 | 3/3 | .081733 | 1.0 | .33333 | .66667 | 1.0 | 7,462.88 |

Candidate-only metrics are not causal deltas. They nevertheless define the
replay signal: candidate component recall and meaningful-program rate are above
c1803, while structure and latency are worse. Training attribution confirms
the objective was active: on the last batch both arms saw 7 component tokens;
raw component CE was 17.5132 for the treatment versus 22.0968 for the control.
Prefix CE stayed near 2.95 and non-component CE near 7.87. The frozen replay
must determine whether lower component CE reproduces as useful generated
component coverage without the structure/latency regressions.

The local explicit-no-sync checkpoint SHA-256 values are `f31ac8fc...2d99`
(control) and `4edbbb7f...678c` (candidate). Neither is reusable, promotable,
synced, or ship evidence. AgentV bundles are complete, fixture gates fail, and
Lean is `not_applicable:screening`.

Machine evidence:
[`autotrain-cycle-1804-component-token-incomplete.json`](autotrain-cycle-1804-component-token-incomplete.json).
