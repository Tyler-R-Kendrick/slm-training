# E472 schema-enum full RICO sharded evaluation — 2026-07-18

E472 refreshes the complete 1,500-row E451 RICO suite after E470–E471 made
component string enums fail closed. It uses contiguous shards because every
process is externally capped at 290 seconds; only normally completed shards
count. Exact-coverage merging is required before any full-RICO or ship claim.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, E468 reference-array semantics, prompt-role
constrained decode, honest constrained slot contracts, eight generation steps,
three attempts, and no fallback.

| Shard | Rows | n | Meaningful | Fidelity | Structure | Recall | Reward | Fail/fallback/timeout |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `[0,96)` | 96 | 1.0 | 1.0 | 0.8852 | 1.0 | 0.9955 | 0 / 0 / 0 |
| 1 | `[96,192)` | 96 | 1.0 | 1.0 | 0.8658 | 1.0 | 0.9936 | 0 / 0 / 0 |
| 2 | `[192,288)` | 96 | 1.0 | 1.0 | 0.8561 | 1.0 | 0.9940 | 0 / 0 / 0 |

Shard 0 completed normally in about 145 seconds and is metric-identical to the
corresponding E460 shard. Its diagnostic AgentV five-gate envelope reports 0/5
because four required suites are absent; this is expected for an isolated
RICO shard and is not a ship result.

Shard 1 completed normally in about 185 seconds. Relative to E460's
corresponding shard, structure improves 0.8652→0.8658 while reward and all
gate metrics are unchanged.

Shard 2 completed normally in about 169 seconds. Structure improves
0.8541→0.8561 versus E460 while reward changes 0.9943→0.9940; all gate metrics
remain perfect with zero failures, fallback, or timeouts.

**Status:** 288/1500 rows complete. No merged or ship claim yet.
