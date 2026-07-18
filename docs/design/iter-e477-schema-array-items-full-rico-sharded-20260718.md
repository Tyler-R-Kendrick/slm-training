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
| 9 | `[864,960)` | 96 | 1.0 | 1.0 | 0.8981 | 1.0 | 0.9932 | 0 / 0 / 0 |
| 10 | `[960,1056)` | 96 | 1.0 | 1.0 | 0.8671 | 1.0 | 0.9932 | 0 / 0 / 0 |
| 11 | `[1056,1152)` | 96 | 1.0 | 1.0 | 0.8669 | 1.0 | 0.9943 | 0 / 0 / 0 |
| 12a | `[1152,1200)` | 48 | 1.0 | 1.0 | 0.8627 | 1.0 | 0.9966 | 0 / 0 / 0 |
| 12b | `[1200,1248)` | 48 | 1.0 | 1.0 | 0.8711 | 1.0 | 0.9929 | 0 / 0 / 0 |
| 13a | `[1248,1296)` | 48 | 1.0 | 1.0 | 0.8878 | 1.0 | 0.9920 | 0 / 0 / 0 |
| 13b | `[1296,1344)` | 48 | 1.0 | 1.0 | 0.8908 | 1.0 | 0.9971 | 0 / 0 / 0 |
| 14a | `[1344,1392)` | 48 | 1.0 | 1.0 | 0.8769 | 1.0 | 0.9958 | 0 / 0 / 0 |
| 14b | `[1392,1440)` | 48 | 1.0 | 1.0 | 0.8530 | 1.0 | 0.9954 | 0 / 0 / 0 |
| 15a | `[1440,1470)` | 30 | 1.0 | 1.0 | 0.8541 | 1.0 | 0.9961 | 0 / 0 / 0 |
| 15b | `[1470,1500)` | 30 | 1.0 | 1.0 | 0.8683 | 1.0 | 0.9954 | 0 / 0 / 0 |

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

Shard 9 completed normally in about 177 seconds and is metric-identical to
E472's corresponding shard, with zero failures, fallback, or timeouts.

Shard 10 completed normally in about 180 seconds and is metric-identical to
E472's corresponding shard, with zero failures, fallback, or timeouts.

Shard 11 completed normally in about 201 seconds and is metric-identical to
E472's corresponding shard, with zero failures, fallback, or timeouts.

The initial 96-row shard-12 attempt
`e477-e396-array-items-rico-shard12-r1` for `[1152,1248)` also reached the
external 290-second ceiling and exited 124 after `KeyboardInterrupt`. It
emitted no evaluation JSON and is excluded from coverage. Remaining 96-row
blocks will use independently capped 48-row replacements.

Replacement shard 12a completed normally in about 158 seconds with
meaningful/fidelity/recall 1.0 and zero failures, fallback, or timeouts.

Replacement shard 12b completed normally in about 221 seconds with the same
perfect core rates and zero reliability counts. The weighted 12a+12b
structure 0.8669 and reward 0.9948 are metric-identical to E472 shard 12.

Replacement shard 13a completed normally in about 155 seconds with
meaningful/fidelity/recall 1.0 and zero reliability counts.

Replacement shard 13b completed normally in about 170 seconds with the same
perfect core rates and zero reliability counts. The weighted 13a+13b
structure 0.8893 and reward 0.9945 are metric-identical to E472 shard 13.

Replacement shard 14a completed normally in about 225 seconds with
meaningful/fidelity/recall 1.0 and zero reliability counts.

Replacement shard 14b completed normally in about 209 seconds with the same
perfect core rates and zero reliability counts. The weighted 14a+14b
structure 0.8650 and reward 0.9956 are metric-identical to E472 shard 14.

The final 60 rows are also split for runtime margin. Shard 15a completed
normally in about 154 seconds with meaningful/fidelity/recall 1.0 and zero
reliability counts.

Shard 15b completed normally in about 115 seconds with the same perfect core
rates and zero reliability counts. The weighted 15a+15b structure 0.8612 and
reward 0.9957 are metric-identical to E472 shard 15.

**Status:** 1500/1500 contiguous rows complete. E478 exact-coverage merge
verified all 21 valid shard files; the two timed-out attempts remain excluded.
