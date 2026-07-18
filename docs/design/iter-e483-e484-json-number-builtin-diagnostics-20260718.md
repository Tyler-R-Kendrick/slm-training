# E483–E484 JSON-number and builtin diagnostics — 2026-07-18

E483 checks the second E479 row containing an invalid byte-spelled Slider
number after E482's fix. E484 checks the remaining audited Slider row, whose
numeric position contains a builtin expression.

Recipe: unchanged E396 checkpoint and E451 RICO offsets 1126 and 801, CPU,
local HF context, 320-token grammar LTR, component-plan weight 2,
slot-component weight 8, schema-enum, array-item, and JSON-number constrained
decode, honest constrained slot contracts, eight generation steps, three
attempts, and no fallback. Both runs completed normally in 3.9 and 6.6 seconds
under separate external 290-second caps.

| Run | Row | Meaningful | Fidelity | Structure | Recall | Reward | Fail/fallback/timeout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E483 | `rico_hf_test_2644` | 1.0 | 1.0 | 0.8250 | 1.0 | 0.9730 | 0 / 0 / 0 |
| E484 | `rico_hf_test_1810` | 1.0 | 1.0 | 0.5103 | 1.0 | 1.0000 | 0 / 0 / 0 |

E483 changes invalid `Slider("n", "continuous", "nnu", 40)` to
schema-valid `Slider("n", "continuous", 0, 100)` without changing metrics.
E484 still emits `Slider("row", "continuous", 40, @Filter())`; builtins
complete as expression type `any`, and the current schema predicate treats
`any` as compatible with the required numeric type.

**Verdict:** E483 confirms the JSON-number fix generalizes. Reject E484's
typed-`any` behavior; `any` expressions should remain legal only when the
expected schema itself is unconstrained.
