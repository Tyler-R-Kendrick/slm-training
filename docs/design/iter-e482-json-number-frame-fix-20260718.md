# E482 JSON-number frame fix — 2026-07-18

E482 reruns E480–E481's RICO row after constraining byte-spelled numeric
frames to valid JSON-number prefixes and exact completions.

Recipe: unchanged E396 checkpoint and E451 RICO row 960
(`rico_hf_test_2249`), CPU, local HF context, 320-token grammar LTR,
component-plan weight 2, slot-component weight 8, schema-enum, array-item, and
JSON-number constrained decode, honest constrained slot contracts, eight
generation steps, three attempts, and no fallback. The run completed normally
in 6.2 seconds under the external 290-second cap.

| n | Meaningful | Fidelity | Structure | Recall | Reward | Fail/fallback/timeout |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.0 | 1.0 | 0.7292 | 1.0 | 0.9970 | 0 / 0 / 0 |

The invalid E480 `Slider("item", "continuous", "nnu", 40)` becomes
`Slider("item", "continuous", -1, 40)`. All Slider numeric positions are now
numbers, while row-level quality metrics remain unchanged.

**Verdict:** accept the generalized JSON-number frame constraint. Diagnostics
for the other audited numeric/builtin rows remain required before broader
evaluation.
