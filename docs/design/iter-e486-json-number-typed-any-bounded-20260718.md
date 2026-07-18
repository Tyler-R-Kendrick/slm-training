# E486 JSON-number and typed-any bounded evaluation — 2026-07-18

E486 evaluates all four bounded suites after E482's JSON-number frame fix and
E485's typed-any schema constraint.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, schema-enum, array-item, JSON-number, and typed-any
constrained decode, prompt-role constrained decode, honest constrained slot
contracts, eight generation steps, three attempts, and no fallback. The run
completed normally in 21.7 seconds under the external 290-second cap.

| Suite | n | Meaningful | Fidelity | Structure | Recall | Reward | Fail/fallback/timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.6822 | 0.6667 | 0.9730 | 0 / 0 / 0 |
| held_out | 5 | 1.0 | 1.0 | 0.7838 | 0.9048 | 0.9868 | 0 / 0 / 0 |
| adversarial | 4 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 | 0 / 0 / 0 |
| ood | 4 | 1.0 | 1.0 | 0.6343 | 0.8750 | 0.9865 | 0 / 0 / 0 |

All metrics are identical to E476. AgentV passes 4/4 with zero execution
errors, and no suite records failures, fallback, or decode timeouts.

**Verdict:** accept bounded evidence. Fresh full RICO remains required, so E479
stays authoritative.
