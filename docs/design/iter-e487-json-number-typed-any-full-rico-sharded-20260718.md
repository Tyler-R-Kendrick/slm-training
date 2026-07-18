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
| 6a | `[288,312)` | 24 | 1.0 | 1.0 | 0.8681 | 1.0 | 0.9921 | 0 / 0 / 0 |
| 6b | `[312,336)` | 24 | 1.0 | 1.0 | 0.9107 | 1.0 | 0.9892 | 0 / 0 / 0 |
| 7a | `[336,360)` | 24 | 1.0 | 1.0 | 0.8932 | 1.0 | 0.9937 | 0 / 0 / 0 |
| 7b | `[360,384)` | 24 | 1.0 | 1.0 | 0.8402 | 1.0 | 0.9950 | 0 / 0 / 0 |
| 8a | `[384,408)` | 24 | 1.0 | 1.0 | 0.8729 | 1.0 | 0.9915 | 0 / 0 / 0 |
| 8b | `[408,432)` | 24 | 1.0 | 1.0 | 0.8482 | 1.0 | 0.9945 | 0 / 0 / 0 |
| 9a | `[432,456)` | 24 | 1.0 | 1.0 | 0.9227 | 1.0 | 0.9891 | 0 / 0 / 0 |
| 9b | `[456,480)` | 24 | 1.0 | 1.0 | 0.8701 | 1.0 | 0.9948 | 0 / 0 / 0 |
| 10a | `[480,504)` | 24 | 1.0 | 1.0 | 0.8852 | 1.0 | 0.9962 | 0 / 0 / 0 |
| 10b | `[504,528)` | 24 | 1.0 | 1.0 | 0.8525 | 1.0 | 0.9912 | 0 / 0 / 0 |
| 11a | `[528,552)` | 24 | 1.0 | 1.0 | 0.8524 | 1.0 | 0.9907 | 0 / 0 / 0 |
| 11b | `[552,576)` | 24 | 1.0 | 1.0 | 0.9088 | 1.0 | 0.9957 | 0 / 0 / 0 |
| 12a | `[576,600)` | 24 | 1.0 | 1.0 | 0.8822 | 1.0 | 0.9920 | 0 / 0 / 0 |
| 12b | `[600,624)` | 24 | 1.0 | 1.0 | 0.8907 | 1.0 | 0.9920 | 0 / 0 / 0 |
| 13a | `[624,640)` | 16 | 1.0 | 1.0 | 0.8863 | 1.0 | 0.9878 | 0 / 0 / 0 |

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

Shards 6a and 6b completed normally in about 130 and 92 seconds. Together they
are prediction-identical to the corresponding E477 rows, with structure
0.8894, reward 0.9907, and zero failures, fallback, or timeouts.

Shards 7a and 7b completed normally in about 104 and 99 seconds. Together they
are prediction-identical to the corresponding E477 rows, with structure
0.8667, reward 0.9944, and zero failures, fallback, or timeouts.

Shards 8a and 8b completed normally in about 105 and 95 seconds. Together they
are prediction-identical to the corresponding E477 rows, with structure
0.8605, reward 0.9930, and zero failures, fallback, or timeouts.

Shards 9a and 9b completed normally in about 105 and 120 seconds. Together
they are prediction-identical to the corresponding E477 rows, with structure
0.8964, reward 0.9919, and zero failures, fallback, or timeouts.

Shards 10a and 10b completed normally in about 142 and 128 seconds. Together
they are prediction-identical to the corresponding E477 rows, with structure
0.8688, reward 0.9938, and zero failures, fallback, or timeouts.

Shards 11a and 11b completed normally in about 99 and 150 seconds. Together
they are prediction-identical to the corresponding E477 rows, with structure
0.8806, reward 0.9932, and zero failures, fallback, or timeouts.

Shard 12a completed normally under the external cap and is
prediction-identical to the corresponding E477 rows, with structure 0.8822,
reward 0.9920, and zero failures, fallback, or timeouts. Two setup attempts
failed before model load or row evaluation because of an incorrect dataset
lookup and an invalid all-suite offset; neither attempt counts as evidence.

Shard 12b completed normally under the external cap and is
prediction-identical to the corresponding E477 rows, with structure 0.8907,
reward 0.9920, and zero failures, fallback, or timeouts.

After row 624, the hard command policy changes to a three-minute total maximum:
interrupt at 170 seconds and force-kill ten seconds later. Future E487 shards
are limited to at most 16 rows; earlier normally completed evidence retains
its historical 290-second policy.

Shard 13a completed normally in about one minute under the new three-minute
policy and is prediction-identical to the corresponding E477 rows, with
structure 0.8863, reward 0.9878, and zero failures, fallback, or timeouts.

**Status:** 640/1500 rows complete. No merged or ship claim yet.
