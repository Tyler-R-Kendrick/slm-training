# E323 corpus-balanced slot-owner objective — 2026-07-17

E323 replaces E322 focal hardness with square-root inverse owner frequency
derived only from E316 training ASTs. The expected per-example weight is
normalized to 1.0. The 22 observed classes receive weights 0.373–3.723.
Architecture and saved parameter shapes match E318; balance power 0.5 and the
persisted weight vector are the only config additions.

The 446-step / 20,044-token CPU scratch run took 122.80s. Checkpoint SHA:
`0df34163083d7959c1de94385d5a2ff984b9ad20ebcff08a11991793f6d5ffdc`.
Weighted/broad NLL are 5.4269/5.4993; loss AgentV passes 1/1. Final-20 raw slot
accuracy falls to 0.6758 versus a 0.6392 batch-majority baseline.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.5464 | 0.6667 | 0.3333 | 0.6407 | Fail recall |
| held_out | 5 | 1.0 | 1.0 | 0.4155 | 0.2000 | 0.1000 | 0.1994 | Fail meaningful/recall |
| adversarial | 4 | 1.0 | 1.0 | 0.5970 | 0.5000 | 0.3750 | 0.4805 | Pass |
| ood | 4 | 1.0 | 1.0 | 0.4304 | 0.5000 | 0.2500 | 0.4992 | Pass |
| limited `rico_held` | 3 | 1.0 | 1.0 | 0.3676 | 1.0000 | 0.5556 | 1.0000 | Pass |

AgentV remains 3/5 with three metric failures. Explicit frequency balancing
does not recover the held-out minority roles.

**Verdict:** reject the checkpoint and do not promote or claim ship. With role
tokens present and minority classes weighted, the remaining missing signal is
ordered local composition; test next-slot context rather than stronger weights.
