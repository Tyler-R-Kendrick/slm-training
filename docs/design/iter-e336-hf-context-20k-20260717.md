# E336 frozen HF context at 20k tokens — 2026-07-17

E336 moves E333's exact choice-tokenizer, component-plan, lexical-prior, and
no-DESIGN recipe from scratch context to frozen
`HuggingFaceTB/SmolLM2-135M`, pinned at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The 795-record E316 training
corpus and leakage-filtered E334 evaluation corpus were reused.

The local CPU train stopped at 421 steps / 20,008 target tokens in 330.36s.
That run completed before the user imposed a non-negotiable five-minute cap
for all subsequent runs. Weighted/broad NLL are 6.2014/6.4501 versus
E333 scratch's 5.4084/5.4961. Loss AgentV passes 5/5. Checkpoint SHA:
`5ce51a406744c58bf5b48c843d329295ffb590921a53e92a9299d50cd6403cbf`.
Checkpoint sync was explicitly disabled for this bounded local validation.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.6111 | 0.4072 | 0.0 | 0.0 | 0.0 |
| held_out | 5 | 1.0 | 0.2233 | 0.2379 | 0.0 | 0.0 | 0.0 |
| adversarial | 4 | 1.0 | 0.2292 | 0.2899 | 0.0 | 0.0 | 0.0 |
| ood | 4 | 1.0 | 0.5917 | 0.2133 | 0.25 | 0.125 | 0.2268 |
| `rico_held` | — | — | — | — | — | — | — |

The evaluation was interrupted immediately when the five-minute policy was
given. Complete JSON exists for the four small suites, but RICO did not finish
and the aggregate ship AgentV bundle was not emitted. This is intentionally
reported as partial evidence, not a full evaluation.

**Verdict:** reject E336. Frozen HF context at the scratch champion's 20k-token
budget regresses loss and collapses meaningful quality on every completed
suite. Do not promote, sync, or claim production/full-RICO readiness. Every
future command must have a hard 300-second wall-clock timeout.

