# E470–E471 schema-enum constrained decode — 2026-07-18

E470 closes a schema-general hole in the choice-codec pushdown state. Component
arguments declared with a string `enum` previously accepted every string
expression. This allowed invalid values such as Callout severity `"itin"` and
the runaway E469 severity literal. The decoder now admits only schema members,
including prefix-constrained byte spelling for members absent from the fixed
literal vocabulary. The vocabulary and E396 checkpoint remain unchanged.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, E468 reference-array semantics, prompt-role
constrained decode, honest constrained slot contracts, eight generation steps,
three attempts, and no fallback. The full smoke and OOD process completed
normally in 16.8 seconds under the external 290-second cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.6822 | 0.6667 | 0.9730 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.7838 | 0.9048 | 0.9868 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.6279 | 0.8750 | 0.9865 |

E471 adds fresh complete held-out and adversarial evidence in a second process
that finished normally in 19.1 seconds. All four bounded suites exactly
preserve E468's metrics with zero failures, fallback, or decode timeouts;
AgentV passes both 2/2 envelopes with zero execution errors. The smoke Callout
changes from invalid `Callout("itin", ...)` to schema-valid
`Callout("info", ...)`; the held-out Slider changes from invalid mode `"in"`
to schema-valid `"continuous"`.

A deterministic replay audit of E468's stored predictions proves this is not a
zero-impact change. It rejects one smoke and one held-out prediction under the
new enum policy, no adversarial or OOD predictions, and 12 of the 1,171
re-encodable RICO predictions. Another 329 stored RICO strings do not
round-trip through the choice encoder, so reuse cannot be certified from the
audit. Fresh held-out, adversarial, and full RICO evidence is required before
updating the five-suite ship result.

**Verdict:** accept the complete bounded schema correction. E468 remains the
authoritative five-suite result until fresh full-RICO evidence completes.
