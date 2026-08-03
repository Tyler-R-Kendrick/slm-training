# Autotrain c1809: balanced container-close is a screening win

**Verdict:** queue for exact fresh-seed confirmation; do not promote. Both
20-step CPU scratch arms completed at 1,608,962 parameters and seed 101809.
The combined zero-parameter objective improves every non-parse smoke quality
metric at a modest latency cost, but `n=3` fixture evidence and absolute gates
are insufficient for promotion or ship claims.

| Arm | Loss | Structure | Binder F1 | Recall | MPR | Fidelity | Reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 0 control | 11.30190 | .0575 | 0 | 0 | 0 | 0 | 0 | 944.03 |
| balance .25 + close 1 | 22.54991 | .174167 | .63333 | .25 | .33333 | .52778 | .76533 | 973.43 |

The candidate avoids the c1807 continuation failure: it emits 24 tokens with
four neural forwards, versus 21 and four for control, rather than c1807's 201
tokens and 51 forwards. Compiler work also improves slightly (`2242.25→2148.00`
ms), while completion states fall `31,483→31,118` and parser forks fall
`32,859→32,126`.

Close alignment received 106 eligible rows. Its margin-violation rate falls
from `.60` on step one to `.20` on the final step. The run also exposed a
harness observability gap: compiler-alignment metrics replaced, rather than
merged with, typed-family attribution. Model harness v300 preserves both metric
families for the exact confirmation; this does not alter the c1809 objective or
evaluation result.

The local explicit-no-sync checkpoint SHA-256 values are `ba99d6c2...e1bf`
(control) and `6a73da29...e81f` (candidate). Neither is reusable, promoted,
synced, or ship evidence. AgentV bundles are complete, fixture gates fail, and
Lean is `not_applicable:screening`. Formal promotion preflight stays locked
until the fresh-seed confirmation establishes a champion.

Machine evidence:
[`autotrain-cycle-1809-balanced-container-close-positive.json`](autotrain-cycle-1809-balanced-container-close-positive.json).
