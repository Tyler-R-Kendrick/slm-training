# E489 E396 JSON-number and typed-any bounded evaluation — 2026-07-18

> Provenance correction: this is branch-only diagnostic evidence. E496 proved
> that current `main` cannot load E396 because the experimental slot-component
> head is absent.

E489 freshly evaluates all four bounded suites under E487's exact generalized
decoder policy. It uses the unchanged E396 checkpoint and E451 corpus, CPU,
local HF context, 320-token grammar LTR, component-plan weight 2,
slot-component weight 8, schema-enum, array-item, JSON-number-frame, and
typed-any constrained decode, honest constrained slot contracts, eight
generation steps, three attempts, and no fallback.

The process completed normally in about 40 seconds under the hard three-minute
policy.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.6822 | 0.6667 | 0.9730 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.7838 | 0.9048 | 0.9868 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.6343 | 0.8750 | 0.9865 |

All four suites pass AgentV with zero execution errors, failures, fallback, or
decode timeouts. Predictions are identical to E476, confirming the new numeric
constraints do not perturb these bounded records.

**Verdict:** accept the bounded evidence for E490 assembly.
