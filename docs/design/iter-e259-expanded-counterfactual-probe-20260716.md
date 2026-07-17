# E259 — expanded depth-stratified counterfactual probe

Date: 2026-07-16
Status: **completed; state-count hypothesis falsified; corpus not admitted for training**

E259 expanded the generalized E258 sampler from four to eight states per
record while keeping every other input fixed: the E228 checkpoint, all 65 E230
document records, strict compiler-tree decoding, seed 258, four candidates per
state, and the independent judge plus meaningful-program/Pareto verifier. The
run started from merged commit
`db1040937208af7fa1b5937f2e2b64472c020c8e` after a fresh fetch/rebase and a
clean `0 behind / 0 ahead` proof.

Trace ID: `fcb8166a47531ec9ba055f155b3ee02e`.

## Measured result

| Measure | Result |
| --- | ---: |
| Accepted document traces | 65 / 65 |
| Exact states replayed | 520 |
| Grammar-legal candidates | 1,528 |
| Independent-judge pass | 403 / 1,528 |
| Fully verified candidates | 97 / 1,528 |
| Qualified events | 38 |
| Qualified decision kinds | 8 |
| Qualified prompt groups | 8 |
| Train / held-out events | 33 / 5 |
| Train / held-out groups | 7 / 1 |
| Set-valued events | 22 |

The probes span 14 compiler-derived decision kinds and all four depth
quartiles. Qualified events add bound-child references and a grammar-right-paren
decision beyond E258. However, every qualified event still comes from the same
eight prompt groups, and all five held-out events belong to `train_cta_01`.

## Decision

Do not admit or persist the E259 export as future training data. Doubling state
count increases events and semantic roles but does not improve prompt-group or
held-out-group support, so the bottleneck is candidate completion quality rather
than state sampling volume. Further state-count scaling would repeat evidence
from the same groups.

The next generalized hypothesis is grammar/AST-aligned state mining from judged
training records: derive exact legal decision states from the parsed gold
trajectory, replay alternatives from those states, and admit only independently
verified good/bad completions. This uses grammar and AST semantics rather than
literal examples, preserves group-derived train/held-out separation, and should
make semantic states available across many more prompt groups.

No checkpoint was written. Machine-readable evidence:
[`quality-matrix-v10-e259-expanded-probe-results.json`](quality-matrix-v10-e259-expanded-probe-results.json).
