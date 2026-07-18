# E330 prompt-conditioned lexeme slot prior — 2026-07-17

E330 is a matched E326 arm that adds pooled prompt context to each
slot-component prediction. The hypothesis is that prompt concepts such as
“hero card” complement local slot-role lexemes.

The 446-step / 20,044-token CPU run took 135.71s. Checkpoint SHA:
`91648e999c09e79a468fbedcab9279da3f9d78fb11dc87fa1030b2c199823d54`.
Weighted/broad NLL are 5.4158/5.5129; loss AgentV passes 1/1. Final-20 slot
accuracy rises from E326's 0.8026 to 0.8947 and loss falls 0.5002→0.3592.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.5464 | 0.6667 | 0.3333 | 0.6407 | Fail recall |
| held_out | 5 | 1.0 | 1.0 | 0.4708 | 0.6000 | 0.3000 | 0.5862 | Pass at floor |
| adversarial | 4 | 1.0 | 1.0 | 0.6304 | 0.7500 | 0.6250 | 0.7238 | Pass |
| ood | 4 | 1.0 | 1.0 | 0.5229 | 1.0000 | 0.5417 | 0.9857 | Pass |
| limited `rico_held` | 3 | 1.0 | 1.0 | 0.4826 | 1.0000 | 0.5556 | 1.0000 | Pass |

**Verdict:** reject E330 and retain E326 as strongest scratch. Better
in-distribution auxiliary accuracy does not improve the final smoke gate and
held-out recall regresses 0.40→0.30. Do not promote or claim ship.
