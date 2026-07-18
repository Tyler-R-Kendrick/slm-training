# E485 typed-any schema fix — 2026-07-18

E485 reruns E484's RICO row after restricting expression type `any` to
unconstrained expected schemas. Typed component arguments now require an exact
primitive, array, object, or component-reference type.

Recipe: unchanged E396 checkpoint and E451 RICO row 801
(`rico_hf_test_1810`), CPU, local HF context, 320-token grammar LTR,
component-plan weight 2, slot-component weight 8, schema-enum, array-item,
JSON-number, and typed-any constrained decode, honest constrained slot
contracts, eight generation steps, three attempts, and no fallback. The run
completed normally in 6.5 seconds under the external 290-second cap.

| n | Meaningful | Fidelity | Structure | Recall | Reward | Fail/fallback/timeout |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.0 | 1.0 | 0.5750 | 1.0 | 1.0000 | 0 / 0 / 0 |

The invalid E484 `Slider("row", "continuous", 40, @Filter())` becomes
schema-valid `Slider("row", "continuous", 40, 40)`. Structure improves
0.5103→0.5750 while meaningful output, fidelity, recall, and reward remain
perfect.

**Verdict:** accept the generalized typed-any constraint. Fresh bounded suites
and full RICO remain required before replacing E479 as authoritative.
