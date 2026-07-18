# E480–E481 Slider numeric-frame diagnostics — 2026-07-18

E480 reproduces a schema-invalid Slider numeric argument found in E479 full
RICO. E481 repeats the same row with read-only raw-choice instrumentation to
separate constrained-token behavior from serialization.

Recipe: unchanged E396 checkpoint and E451 RICO row 960
(`rico_hf_test_2249`), CPU, local HF context, 320-token grammar LTR,
component-plan weight 2, slot-component weight 8, schema-enum and array-item
constraints, honest constrained slot contracts, eight generation steps, three
attempts, and no fallback. E480 and E481 completed normally in 6.3 and 5.4
seconds under separate external 290-second caps.

| Run | n | Meaningful | Fidelity | Structure | Recall | Reward | Fail/fallback/timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E480 reproduction | 1 | 1.0 | 1.0 | 0.7292 | 1.0 | 0.9970 | 0 / 0 / 0 |
| E481 raw-choice trace | 1 | 1.0 | 1.0 | 0.7292 | 1.0 | 0.9970 | 0 / 0 / 0 |

Both runs emit `Slider("item", "continuous", "nnu", 40)`, although Slider
argument 2 requires a number. E481 shows the raw stream uses
`LIT_NUM B:6e B:6e B:75 LIT_END`: the constrained state accepts arbitrary
non-empty bytes in a numeric frame, then the serializer quotes the invalid
numeric payload as `"nnu"`. This is a deterministic codec bug, not a model
training deficit.

**Verdict:** reject the numeric-frame behavior. Constrain byte-spelled numbers
to valid JSON-number prefixes and exact completions, then rerun this row before
any broader evaluation.
