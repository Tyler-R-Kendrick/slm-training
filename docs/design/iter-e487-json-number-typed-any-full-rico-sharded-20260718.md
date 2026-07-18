# E487 JSON-number and typed-any full RICO sharded evaluation — 2026-07-18

E487 refreshes the complete 1,500-row E451 RICO suite after E482 constrained
byte-spelled JSON numbers and E485 rejected `any` expressions in typed schemas.
Every shard is limited to at most 48 rows and externally capped at 290 seconds;
only normally completed shards count. Canonical exact-coverage merging is
required before a full-RICO or ship claim.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, schema-enum, array-item, JSON-number, and typed-any
constrained decode, prompt-role constrained decode, honest constrained slot
contracts, eight generation steps, three attempts, and no fallback.

| Shard | Rows | n | Meaningful | Fidelity | Structure | Recall | Reward | Fail/fallback/timeout |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `[0,48)` | 48 | 1.0 | 1.0 | 0.9052 | 1.0 | 0.9959 | 0 / 0 / 0 |
| 1 | `[48,96)` | 48 | 1.0 | 1.0 | 0.8652 | 1.0 | 0.9951 | 0 / 0 / 0 |
| 2 | `[96,144)` | 48 | 1.0 | 1.0 | 0.8572 | 1.0 | 0.9936 | 0 / 0 / 0 |
| 3 | `[144,192)` | 48 | 1.0 | 1.0 | 0.8744 | 1.0 | 0.9935 | 0 / 0 / 0 |
| 4a | `[192,216)` | 24 | 1.0 | 1.0 | 0.8926 | 1.0 | 0.9930 | 0 / 0 / 0 |
| 4b | `[216,240)` | 24 | 1.0 | 1.0 | 0.8187 | 1.0 | 0.9932 | 0 / 0 / 0 |
| 5a | `[240,264)` | 24 | 1.0 | 1.0 | 0.8631 | 1.0 | 0.9949 | 0 / 0 / 0 |
| 5b | `[264,288)` | 24 | 1.0 | 1.0 | 0.8501 | 1.0 | 0.9947 | 0 / 0 / 0 |

Shard 0 completed normally in about 177 seconds and is metric-identical to the
corresponding E477 rows, with zero failures, fallback, or timeouts.

Shard 1 completed normally in about 199 seconds and is metric-identical to the
corresponding E477 rows, with zero failures, fallback, or timeouts.

Shard 2 completed normally in about 163 seconds and is metric-identical to the
corresponding E477 rows, with zero failures, fallback, or timeouts.

Shard 3 completed normally in about 159 seconds and is metric-identical to the
corresponding E477 rows, with zero failures, fallback, or timeouts.

The first `[192,240)` attempt was interrupted by the external 290-second cap.
It is excluded from evidence. Replacement shards 4a and 4b completed normally
in about 128 and 125 seconds, respectively. Together they are
prediction-identical to the corresponding E477 rows, with structure 0.8557,
reward 0.9931, and zero failures, fallback, or timeouts.

Shards 5a and 5b completed normally in about 141 and 120 seconds. Together
they are prediction-identical to the corresponding E477 rows, with structure
0.8566, reward 0.9948, and zero failures, fallback, or timeouts.

**Status:** 288/1500 rows complete. No merged or ship claim yet.
