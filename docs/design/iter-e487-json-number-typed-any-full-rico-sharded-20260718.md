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
| 13b | `[640,656)` | 16 | 1.0 | 1.0 | 0.8974 | 1.0 | 0.9938 | 0 / 0 / 0 |
| 13c | `[656,672)` | 16 | 1.0 | 1.0 | 0.8523 | 1.0 | 0.9991 | 0 / 0 / 0 |
| 14a | `[672,688)` | 16 | 1.0 | 1.0 | 0.8525 | 1.0 | 0.9948 | 0 / 0 / 0 |
| 14b | `[688,704)` | 16 | 1.0 | 1.0 | 0.8573 | 1.0 | 0.9961 | 0 / 0 / 0 |
| 14c | `[704,720)` | 16 | 1.0 | 1.0 | 0.9107 | 1.0 | 0.9906 | 0 / 0 / 0 |
| 15a | `[720,736)` | 16 | 1.0 | 1.0 | 0.9102 | 1.0 | 0.9957 | 0 / 0 / 0 |
| 15b | `[736,752)` | 16 | 1.0 | 1.0 | 0.8433 | 1.0 | 0.9888 | 0 / 0 / 0 |
| 15c | `[752,768)` | 16 | 1.0 | 1.0 | 0.8740 | 1.0 | 0.9933 | 0 / 0 / 0 |
| 16a | `[768,784)` | 16 | 1.0 | 1.0 | 0.8784 | 1.0 | 0.9934 | 0 / 0 / 0 |
| 16b | `[784,800)` | 16 | 1.0 | 1.0 | 0.8178 | 1.0 | 0.9946 | 0 / 0 / 0 |
| 16c | `[800,816)` | 16 | 1.0 | 1.0 | 0.8441 | 1.0 | 0.9972 | 0 / 0 / 0 |
| 17a | `[816,832)` | 16 | 1.0 | 1.0 | 0.8747 | 1.0 | 0.9948 | 0 / 0 / 0 |
| 17b | `[832,848)` | 16 | 1.0 | 1.0 | 0.9148 | 1.0 | 0.9957 | 0 / 0 / 0 |
| 17c | `[848,864)` | 16 | 1.0 | 1.0 | 0.8544 | 1.0 | 0.9918 | 0 / 0 / 0 |
| 18a | `[864,880)` | 16 | 1.0 | 1.0 | 0.9127 | 1.0 | 0.9901 | 0 / 0 / 0 |
| 18b | `[880,896)` | 16 | 1.0 | 1.0 | 0.8747 | 1.0 | 0.9957 | 0 / 0 / 0 |
| 18c | `[896,912)` | 16 | 1.0 | 1.0 | 0.9199 | 1.0 | 0.9916 | 0 / 0 / 0 |

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

Shard 13b completed normally under the three-minute policy and is
prediction-identical to the corresponding E477 rows, with structure 0.8974,
reward 0.9938, and zero failures, fallback, or timeouts.

Shard 13c completed normally in about 68 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.8523, reward 0.9991, and zero failures, fallback, or timeouts.

Shard 14a completed normally in about 68 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.8525, reward 0.9948, and zero failures, fallback, or timeouts.

Shard 14b completed normally in about 64 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.8573, reward 0.9961, and zero failures, fallback, or timeouts.

Shard 14c completed normally in about 60 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.9107, reward 0.9906, and zero failures, fallback, or timeouts.

Shard 15a completed normally in about 77 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.9102, reward 0.9957, and zero failures, fallback, or timeouts.

Shard 15b completed normally in about 69 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.8433, reward 0.9888, and zero failures, fallback, or timeouts.

Shard 15c completed normally in about 69 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.8740, reward 0.9933, and zero failures, fallback, or timeouts.

Shard 16a completed normally in about 90 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.8784, reward 0.9934, and zero failures, fallback, or timeouts.

Shard 16b completed normally in about 67 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.8178, reward 0.9946, and zero failures, fallback, or timeouts.

Shard 16c completed normally in about 61 seconds under the three-minute policy
with structure 0.8441, reward 0.9972, and zero failures, fallback, or timeouts.
This is the first non-identical E487 slice: one of 16 predictions changed
(`rico_hf_test_1810`). The JSON-number constraint replaced E477's typed
`@Filter()` value in a numeric slider slot with `40`; row structure improved
from 0.5103 to 0.5750 and slice structure improved by 0.00404, while fidelity,
type recall, and reward remained unchanged. This is positive activation
evidence, not a full-suite claim.

Shard 17a completed normally in about 60 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.8747, reward 0.9948, and zero failures, fallback, or timeouts.

Shard 17b completed normally in about 70 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.9148, reward 0.9957, and zero failures, fallback, or timeouts.

Shard 17c completed normally in about 77 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.8544, reward 0.9918, and zero failures, fallback, or timeouts.

Shard 18a completed normally in about 57 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.9127, reward 0.9901, and zero failures, fallback, or timeouts.

Shard 18b completed normally in about 78 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.8747, reward 0.9957, and zero failures, fallback, or timeouts.

Shard 18c completed normally in about 76 seconds under the three-minute policy
and is prediction-identical to the corresponding E477 rows, with structure
0.9199, reward 0.9916, and zero failures, fallback, or timeouts.

**Status:** 912/1500 rows complete. No merged or ship claim yet.
