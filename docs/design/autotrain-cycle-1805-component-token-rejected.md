# Autotrain c1805: component-token arm is rejected

**Verdict:** reproduced runtime-specific unblock, absolute quality reject. This
cycle re-evaluated the exact frozen c1804 checkpoints at the same seed and
policy. The control again timed out on all 3 smoke records; the component-token
candidate again completed all 3. That repeated asymmetry is a real runtime
signal, but it is not a model-quality win: candidate structure is `.081733`,
below the `.35` fixture gate, and p50 remains `7,186.02` ms.

| Arm | Train source | Complete | Structure | Binder F1 | Recall | MPR | Fidelity | p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 0 control | frozen c1804 | 0/3 | — | — | — | — | — | — |
| component weight 1 | frozen c1804 | 3/3 | .081733 | 1.0 | .33333 | .66667 | 1.0 | 7,186.02 |

The candidate checkpoint SHA-256 is `4edbbb7f...678c`; the control is
`f31ac8fc...2d99`. No new checkpoint was trained in c1805. Both remain local,
explicit-no-sync scratch artifacts and are rejected for reuse, promotion, and
shipping. AgentV bundles were emitted for both arms. The control's repeated
typed decode timeouts make a paired quality delta unavailable, while the
candidate's absolute quality failure is sufficient to retire the approach.

The next distinct, size-matched hypothesis directly weights grammar `STRUCT`
tokens. It targets scaffold formation rather than retuning the rejected
component-token dose, adds no parameters, and does not change the grammar
domain or constrained decoder. Lean is `not_applicable:retry_measurement` for
this fixture replay; both Lean CI lanes still gate the harness delivery.

Machine evidence:
[`autotrain-cycle-1805-component-token-rejected.json`](autotrain-cycle-1805-component-token-rejected.json).
