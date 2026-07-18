# E490 E396 JSON-number and typed-any full ship gates — 2026-07-18

E490 assembles E489's fresh bounded suites with E488's exact 1,500-row RICO
merge under one unchanged E396 checkpoint and evaluation policy.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, component-plan weight 2, slot-component weight 8,
schema-enum, array-item, JSON-number-frame, and typed-any constrained decode,
prompt-role constrained decode, honest constrained slot contracts, eight
generation steps, three attempts, and no fallback. The assembly completed
normally in about two seconds under the hard three-minute policy.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.6822 | 0.6667 | 0.9730 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.7838 | 0.9048 | 0.9868 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.6343 | 0.8750 | 0.9865 |
| rico_held | 1500 | 1.0 | 1.0 | 1.0 | 0.8737 | 1.0 | 0.9939 |

All five current-policy gates pass with zero failures, fallback, or decode
timeouts; AgentV passes 5/5 with zero execution errors. Relative to E479, the
four bounded suites are prediction-identical. Full-RICO structure improves by
0.0000431 while reward remains unchanged within floating-point precision.

The exact checkpoint is durable at
`hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1`.
This satisfies checkpoint persistence; it does not change the serving
deployment or constitute a new checkpoint promotion.

E496 later loaded the same checkpoint SHA on clean current `main`. Its honest
smoke result retained syntax parse 1.0 but fell to meaningful 0.0, fidelity
0.5556, structure 0.1131, type recall 0.0, and AgentV 0/5. The experimental
decoder branch used here was never reconciled into `main`, and this result did
not record an exact code revision.

**Verdict:** retain E490 as branch-only diagnostic evidence for the durable E396
checkpoint. It is not current-main or deployable-code evidence; E496 is the
authoritative current-main audit. Serving promotion remains unchanged.
