# E322 focal slot-owner objective — 2026-07-17

E322 tests the E321 imbalance diagnosis with focal gamma 2. The corrected E318
diffusion recipe, E316 corpus, seed, architecture, 20k-token budget, slot-only
representation, and E319/E320 decoder are fixed. Saved state keys/shapes and
449,301 parameters match E318; gamma is the only config delta.

The CPU scratch run stopped at 446 steps / 20,044 target tokens in 144.22s.
Checkpoint SHA:
`2d69cbfd710e16d939b8bba53e09f3d426cf1deb67384ba77fd9a9618e7e8507`.
Weighted/broad NLL are 5.4247/5.4988; loss AgentV passes 1/1. Final-20 slot
loss is 0.9952, raw accuracy 0.7050, and batch-majority baseline 0.6392.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.5464 | 0.6667 | 0.3333 | 0.6407 | Fail recall |
| held_out | 5 | 1.0 | 1.0 | 0.4011 | 0.2000 | 0.1000 | 0.1994 | Fail meaningful/recall |
| adversarial | 4 | 1.0 | 1.0 | 0.5970 | 0.5000 | 0.3750 | 0.4805 | Pass |
| ood | 4 | 1.0 | 1.0 | 0.4304 | 0.5000 | 0.2500 | 0.4992 | Pass |
| limited `rico_held` | 3 | 1.0 | 1.0 | 0.3850 | 1.0000 | 0.5556 | 1.0000 | Pass |

AgentV remains 3/5, but metric failures increase two→three. Against E319/E320,
held-out meaningful/recall regress 0.40/0.20→0.20/0.10.

**Verdict:** reject focal loss and do not promote or claim ship. Focal hardness
does not directly correct the 22-way owner-frequency imbalance; the next arm
should use corpus-derived class weights.
