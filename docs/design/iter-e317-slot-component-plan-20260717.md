# E317 slot-conditioned component plan — 2026-07-17

E317 tests whether the remaining E316 component-role errors can be corrected by
a learned, decision-local component head. The head maps each visible slot plus
the prompt representation to its direct containing component. It is trained
from parsed targets and biases only compiler-legal component choices for the
next unconsumed visible slot. No evaluation IDs, literal evaluation-slot rules,
or alternate decode path were added.

The matched CPU scratch run stopped at 446 steps / 20,044 target tokens in
144.57 seconds. It used the E316 semantic-slot corpus, seed 0, the E311
token-pooled global component plan, and slot loss/decode weights 1.0.
Checkpoint SHA:
`42476f4ccf97adf1249981eee9481ec81c3816af320c71747299144c2734e130`.
It is a local scratch artifact with explicit `--no-sync-checkpoints`.

## Training diagnostics

| Measure | E316 | E317 |
| --- | ---: | ---: |
| Trainable parameters | 402,012 | 405,717 |
| Weighted NLL | **5.4155** | 5.4483 |
| Broad NLL | **5.4832** | 5.5233 |
| Final-20 global plan loss | 1.9185 | **1.9176** |
| Global root accuracy | 0.9500 | 0.9500 |
| Global bound top-k recall | 0.4621 | 0.4621 |
| Global bound-count MAE | **0.2794** | 0.2829 |
| Slot-component loss | — | 1.2056 |
| Slot-component accuracy | — | 0.7008 |

Loss-suite AgentV passes 1/1. The head learns its training task, but NLL and the
global plan remain effectively unchanged.

## Honest five-suite result

The intended intervention uses slot decode weight 1.0 under the frozen E315
policy: honest visible contract, tree compiler, distinct-slot auto content
floor, and no unconstrained fallback.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Component recall | Reward | Slot changes | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.5464 | 0.6667 | 0.3333 | 0.6407 | 2 | Fail: recall needs 0.35 |
| held_out | 5 | 1.0 | 1.0 | 0.4011 | 0.2000 | 0.1000 | 0.1994 | 6 | Fail: meaningful and recall |
| adversarial | 4 | 1.0 | 1.0 | 0.5970 | 0.5000 | 0.3750 | 0.4805 | 4 | Pass |
| ood | 4 | 1.0 | 1.0 | 0.4304 | 0.5000 | 0.2500 | 0.4992 | 4 | Pass |
| limited `rico_held` | 3 | 1.0 | 1.0 | 0.5350 | 1.0000 | 0.5556 | 1.0000 | 12 | Pass |

The intended arm has three metric failures and AgentV 3/5. Relative to E316,
held-out meaningful/recall fall from 0.40/0.20 to 0.20/0.10 and OOD
meaningful/recall fall from 1.0/0.5417 to 0.50/0.25. Parse and visible-slot
fidelity remain 1.0 everywhere.

## Frozen-checkpoint decode ablation

To separate learned representation effects from decode intervention, the same
checkpoint was evaluated at weights 0, 0.25, 0.5, and 1.0.

| Weight | Choice changes | Failures | AgentV | Held meaningful / recall | OOD meaningful / recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 2 | 3/5 | 0.40 / 0.20 | **1.00 / 0.5417** |
| 0.25 | 15 | 2 | 3/5 | 0.40 / 0.20 | 0.50 / 0.25 |
| 0.5 | 24 | 3 | 3/5 | 0.20 / 0.10 | 0.50 / 0.25 |
| 1.0 | 28 | 3 | 3/5 | 0.20 / 0.10 | 0.50 / 0.25 |

Weight 0 reproduces E316 suite metrics exactly. The first nonzero weight changes
choices without clearing either remaining gate and sharply regresses OOD; larger
weights also regress held-out quality. This falsifies the useful-decode
hypothesis for this slot-only representation at the tested budget.

**Verdict:** reject the E317 slot-component decode mechanism and do not promote
or claim ship. Retain E316 as the strongest scratch candidate. The next lever
must add compositional role evidence that generalizes to unseen slot names
rather than increase or retune this head's decode weight.
