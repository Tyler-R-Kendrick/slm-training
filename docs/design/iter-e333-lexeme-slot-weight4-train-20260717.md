# E333 persisted lexical slot weight 4 — 2026-07-17

E333 reruns E326's exact 20k-token recipe with slot-component decode weight 4
persisted in checkpoint metadata. No evaluation override is needed.

The 446-step / 20,044-token CPU run took 123.81s. Checkpoint SHA:
`ca6c290d2617f31c4c5eae6061690974e62d5232252d83d583d03310b4eca9a1`.
Weighted/broad NLL are 5.4084/5.4961; loss AgentV passes 1/1. Final-20 slot
accuracy is 0.8026 versus majority baseline 0.6184.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.6281 | 0.6667 | 0.5000 | 0.6407 | Pass |
| held_out | 5 | 1.0 | 1.0 | 0.5451 | 0.6000 | 0.4000 | 0.5862 | Pass |
| adversarial | 4 | 1.0 | 1.0 | 0.6304 | 0.7500 | 0.6250 | 0.7238 | Pass |
| ood | 4 | 1.0 | 1.0 | 0.6213 | 0.7500 | 0.5625 | 0.7425 | Pass |
| limited `rico_held` | 3 | 1.0 | 1.0 | 0.6717 | 1.0000 | 0.5556 | 1.0000 | Pass |

AgentV passes 5/5 with no execution errors and all current scratch gates pass.

**Verdict:** promote E333 as the local scratch champion. Do not claim
production ship: this run explicitly disabled checkpoint sync, used scratch
context, and evaluated only 19 examples including limited `rico_held` n=3.
Next validation must broaden RICO/HF evidence rather than tune these gates.
