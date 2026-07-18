# E479 E396 schema-array item full ship gates — 2026-07-18

E479 assembles E476's fresh bounded suites with E478's exact 1,500-row RICO
merge under one unchanged E396 checkpoint and evaluation policy. The
schema-general decoder now enforces pinned item schemas inside arrays,
preventing component references such as `Button` where `AreaChart` requires
`Series`.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, schema-enum and array-item constrained decode,
prompt-role constrained decode, honest constrained slot contracts, eight
generation steps, three attempts, and no fallback. Every accepted generating
process completed normally under the external 290-second cap; two timed-out
96-row attempts emitted no evaluation JSON and were excluded. The five-suite
assembly completed normally in about one second under the same cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.6822 | 0.6667 | 0.9730 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.7838 | 0.9048 | 0.9868 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.6343 | 0.8750 | 0.9865 |
| rico_held | 1500 | 1.0 | 1.0 | 1.0 | 0.8736 | 1.0 | 0.9939 |

All five local ship gates pass with zero failures, fallback, or decode
timeouts; AgentV passes 5/5 with zero execution errors. Relative to E474,
smoke, held-out, adversarial, and full-RICO aggregate metrics are unchanged;
OOD structure improves 0.6279→0.6343 at unchanged recall and reward. The
audited invalid RICO `AreaChart(["…"], [Button])` output is now schema-valid
`AreaChart([], [])`.

**Verdict:** accept E479 as the current-policy local ship-gate evidence for the
unchanged E396 champion. This is not a production HF ship claim; durable bucket
sync remains pending.
