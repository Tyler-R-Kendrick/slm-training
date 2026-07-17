# E316 semantic slot-role matched train — 2026-07-17

E316 trains the E311 token-pooled component-plan recipe on the accepted
semantic-slot corpus and evaluates it with the accepted E315 distinct-slot
floor. The sole intended intervention is the deterministic generation-slot
augmentation; seed, architecture, objective, 20k target-token budget, honest
policy, and five suites remain fixed.

The run stopped at 446 steps / 20,044 target tokens in 125.65 seconds.
Checkpoint SHA:
`b2f6e676363ac8ce7690fff0a2d56ec276d72013ba4e4a2f0083e1315c11395a`.
It is a local CPU scratch artifact with explicit `--no-sync-checkpoints`.
The expanded scratch prompt vocabulary raises trainable parameters from
399,836 to 402,012, so capacity is close but not identical.

## Training diagnostics

| Measure | E314 visible contract | E316 semantic slots |
| --- | ---: | ---: |
| Weighted NLL | **5.0561** | 5.4155 |
| Broad NLL | **5.2258** | 5.4832 |
| Final-20 plan loss | 2.3308 | **1.9185** |
| Root accuracy | 0.8500 | **0.9500** |
| Bound top-k recall | 0.4104 | **0.4621** |
| Bound-count MAE | 0.3416 | **0.2794** |

Loss-suite AgentV passes 1/1. The worse NLL is expected evidence that the fixed
budget has not fully absorbed the larger, broader corpus; it does not override
the causal semantic evaluation.

## Honest five-suite result

| Suite | n | Parse | Fidelity | Structure | Meaningful | Component recall | Reward | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.5464 | 0.6667 | 0.3333 | 0.6407 | Fail: recall needs 0.35 |
| held_out | 5 | 1.0 | 1.0 | 0.4431 | 0.4000 | 0.2000 | 0.3916 | Fail: recall needs 0.30 |
| adversarial | 4 | 1.0 | 1.0 | 0.4453 | 0.5000 | 0.3750 | 0.4865 | Pass |
| ood | 4 | 1.0 | 1.0 | 0.5104 | 1.0000 | 0.5417 | 0.9857 | Pass |
| limited `rico_held` | 3 | 1.0 | 1.0 | 0.3369 | 1.0000 | 0.5556 | 1.0000 | Pass |

Against E315, failures fall 5→2 and AgentV improves 2/5→3/5. Most notably,
held-out meaningful rate and component recall move from 0/0 to 0.40/0.20; OOD
moves from 0.25/0.125 to 1.0/0.5417; and limited RICO reaches reward 1.0.
Slot fidelity and parse remain 1.0 everywhere with no fallback.

**Verdict:** retain E316 as the strongest current scratch candidate and retain
the semantic-slot data lever. Do not promote or claim ship: smoke narrowly
misses component recall and held-out recall remains 0.10 below its gate. The
next experiment should target the remaining per-record component-role errors,
not weaken thresholds or scale the existing global plan bias.
