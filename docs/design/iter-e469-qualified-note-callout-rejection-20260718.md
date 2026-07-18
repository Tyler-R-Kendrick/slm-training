# E469 qualified-note Callout rejection — 2026-07-18

E469 tests whether qualified UI-note phrases should share the explicit
Callout prompt contract. The E357/E451 corpus contains 15 prompts with a
standalone `note`, all paired with Callout gold; the proposed decoder pattern
conservatively covers only counted/article-prefixed `supporting`,
`informational`, `warning`, or `error` notes.

Recipe: unchanged E396 checkpoint and E451 OOD suite, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, E468 reference-array fail-closed semantics, honest
constrained slot contracts, eight generation steps, three attempts, and no
fallback. The complete four-row OOD process finished normally in about ten
seconds under the external 290-second cap.

| Policy | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E468 accepted | 4 | 1.0 | 1.0 | 1.0 | 0.6279 | 0.8750 | 0.9865 |
| E469 qualified note | 4 | 1.0 | 0.7500 | 0.8333 | 0.6118 | 0.6875 | 0.7365 |

The new constraint collapses `ood_gallery_01` into a root Callout with a long
invalid severity literal. That row has meaningful failure
`low_component_recall:0.25`, fidelity 0.3333, structure 0.1900, type recall
0.0, and reward 0.0. Aggregate OOD has one failure; fallback and decode
timeout counts remain zero. The single-suite AgentV envelope reports 1/5
because four required suites are absent.

**Verdict:** reject and revert the qualified-note mapping. A corpus-level
association is insufficient when the component constraint destabilizes
grammar-role selection. E468 remains the current-policy five-suite result.
