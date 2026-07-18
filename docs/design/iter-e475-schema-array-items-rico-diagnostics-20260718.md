# E475 schema-array item constraints — 2026-07-18

E475 closes another schema-general hole in the choice-codec pushdown state.
Component arguments declared as arrays previously retained only the outer
`array` type and discarded their `items` schema. E474 therefore emitted two
RICO `AreaChart` programs whose series arrays contained `Button` references
even though the pinned schema requires `Series`.

The decoder now carries each array's item schema through nested variadic frames
and validates every completed element. It applies to primitive, component
reference, union, and nested-array item schemas without changing the vocabulary
or E396 checkpoint.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, E474 policy, honest constrained slot contracts, eight
generation steps, three attempts, and no fallback. The first affected-row
process completed normally in 13.2 seconds under the external 290-second cap.

| Offset | ID | Before | After | Structure | Recall | Reward |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 368 | `rico_hf_test_811` | `AreaChart([...], [Button])` | `AreaChart([], [])` | 0.6913 | 1.0 | 0.9700 |
| 781 | `rico_hf_test_1773` | `AreaChart([...], [Button])` | `AreaChart([], [])` | 0.6913 | 1.0 | 0.9700 |

The repaired row preserves parse, meaningful output, fidelity, structure,
component recall, and reward with zero failures, fallback, or decode timeouts,
while removing the schema-invalid series element. Its diagnostic AgentV
five-gate envelope reports 0/5 because the other suites are absent.

The second affected-row process completed normally in 12.7 seconds with the
same schema correction and metric preservation.

A deterministic replay audit proves the lever is broad. It newly rejects two
held-out and one OOD stored prediction, no smoke or adversarial predictions,
and 195 of the 1,172 re-encodable RICO predictions. Another 328 stored RICO
strings do not round-trip through the choice encoder, so reuse cannot be
certified.

**Verdict:** accept the two-row diagnostic correction, but require fresh
bounded and full-RICO evidence before any gate promotion. E474 remains
authoritative meanwhile.
