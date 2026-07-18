# E477 schema-array item full RICO sharded evaluation — 2026-07-18

E477 refreshes the complete 1,500-row E451 RICO suite after E475 began
enforcing pinned array-item schemas. It uses contiguous standalone shards
because every process is externally capped at 290 seconds; only normally
completed shards count. Canonical exact-coverage merging is required before a
full-RICO or ship claim.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, schema-enum and array-item constrained decode,
prompt-role constrained decode, honest constrained slot contracts, eight
generation steps, three attempts, and no fallback.

| Shard | Rows | n | Meaningful | Fidelity | Structure | Recall | Reward | Fail/fallback/timeout |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `[0,96)` | 96 | 1.0 | 1.0 | 0.8852 | 1.0 | 0.9955 | 0 / 0 / 0 |
| 1 | `[96,192)` | 96 | 1.0 | 1.0 | 0.8658 | 1.0 | 0.9936 | 0 / 0 / 0 |
| 2 | `[192,288)` | 96 | 1.0 | 1.0 | 0.8561 | 1.0 | 0.9940 | 0 / 0 / 0 |
| 3 | `[288,384)` | 96 | 1.0 | 1.0 | 0.8781 | 1.0 | 0.9925 | 0 / 0 / 0 |
| 4 | `[384,480)` | 96 | 1.0 | 1.0 | 0.8785 | 1.0 | 0.9925 | 0 / 0 / 0 |
| 5 | `[480,576)` | 96 | 1.0 | 1.0 | 0.8747 | 1.0 | 0.9935 | 0 / 0 / 0 |
| 6 | `[576,672)` | 96 | 1.0 | 1.0 | 0.8826 | 1.0 | 0.9928 | 0 / 0 / 0 |
| 7 | `[672,768)` | 96 | 1.0 | 1.0 | 0.8747 | 1.0 | 0.9932 | 0 / 0 / 0 |
| 8a | `[768,816)` | 48 | 1.0 | 1.0 | 0.8454 | 1.0 | 0.9951 | 0 / 0 / 0 |
| 8b | `[816,864)` | 48 | 1.0 | 1.0 | 0.8813 | 1.0 | 0.9941 | 0 / 0 / 0 |

Shard 0 completed normally in about 150 seconds and is metric-identical to
E472's corresponding enum-constrained shard. Its diagnostic AgentV five-gate
envelope reports 0/5 because four required suites are absent; this is expected
for an isolated RICO shard and is not a ship result.

Shard 1 completed normally in about 181 seconds and is metric-identical to
E472's corresponding shard, with zero failures, fallback, or timeouts.

Shard 2 completed normally in about 174 seconds and is metric-identical to
E472's corresponding shard, with zero failures, fallback, or timeouts.

Shard 3 completed normally in about 180 seconds and is metric-identical to
E472's corresponding shard, with zero failures, fallback, or timeouts.

Shard 4 completed normally in about 180 seconds and is metric-identical to
E472's corresponding shard, with zero failures, fallback, or timeouts.

Shard 5 completed normally in about 202 seconds and is metric-identical to
E472's corresponding shard, with zero failures, fallback, or timeouts.

Shard 6 completed normally in about 178 seconds and is metric-identical to
E472's corresponding shard, with zero failures, fallback, or timeouts.

Shard 7 completed normally in about 210 seconds and is metric-identical to
E472's corresponding shard, with zero failures, fallback, or timeouts.

The initial 96-row shard-8 attempt
`e477-e396-array-items-rico-shard08-r1` for `[768,864)` reached the external
290-second ceiling and exited 124 after `KeyboardInterrupt`. It emitted no
evaluation JSON, is excluded from coverage, and is being replaced by smaller
independently capped shards.

Replacement shard 8a completed normally in about 108 seconds. On audited row
`rico_hf_test_1773`, the invalid E472 `AreaChart(["…"], [Button])` becomes
schema-valid `AreaChart([], [])`; aggregate meaningful/fidelity/recall remain
1.0 with zero failures, fallback, or timeouts.

Replacement shard 8b completed normally in about 105 seconds, also with
meaningful/fidelity/recall 1.0 and zero reliability counts. The weighted
8a+8b structure 0.8634 and reward 0.9946 are metric-identical to E472 shard 8
despite the schema-corrected audited output.

**Status:** 864/1500 rows complete. No merged or ship claim yet.
