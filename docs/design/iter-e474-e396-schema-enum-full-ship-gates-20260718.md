# E474 E396 schema-enum full ship gates — 2026-07-18

E474 assembles the fresh E470–E471 bounded suites with E473's exact 1,500-row
RICO merge under one unchanged E396 checkpoint and evaluation policy. The
schema-general decoder constraint permits only pinned string-enum members,
correcting invalid Callout severity and Slider mode emissions without changing
the tokenizer vocabulary or checkpoint weights.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, E468 reference-array semantics, schema-enum
constrained decode, prompt-role constrained decode, honest constrained slot
contracts, eight generation steps, three attempts, and no fallback. Every
generating process completed normally under the external 290-second cap; the
five-suite assembly completed normally in 1.8 seconds under the same cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.6822 | 0.6667 | 0.9730 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.7838 | 0.9048 | 0.9868 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.6279 | 0.8750 | 0.9865 |
| rico_held | 1500 | 1.0 | 1.0 | 1.0 | 0.8736 | 1.0 | 0.9939 |

All five local ship gates pass with zero failures, fallback, or decode
timeouts; AgentV passes 5/5 with zero execution errors. Relative to E468, the
four bounded suite metrics are unchanged and full-RICO structure changes
0.8740→0.8736 at effectively unchanged reward 0.9939. The output is more
schema-correct: smoke Callout severity changes `"itin"`→`"info"` and held-out
Slider mode changes `"in"`→`"continuous"`.

**Verdict:** accept E474 as the current-policy local ship-gate evidence for the
unchanged E396 champion. This is not a production HF ship claim; durable bucket
sync remains pending.
