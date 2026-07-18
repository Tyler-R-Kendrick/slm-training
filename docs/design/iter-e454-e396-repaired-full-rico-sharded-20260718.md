# E454 E396 repaired full-RICO sharded evaluation — 2026-07-18

E454 expands E453 across E451's repaired 1,500-row RICO suite. Every shard
uses the unchanged E396 checkpoint, CPU, local HF context, 320-token grammar
LTR, automatic content floor, component-plan weight 2, slot-component weight
8, prompt-role constrained decode, honest constrained slot contracts, eight
generation steps, three attempts, and no unconstrained fallback.

Each process is externally capped at 290 seconds with a ten-second forced kill.
Interrupted or timed-out shards are excluded rather than merged.

| Shard | Rows | n | Meaningful | Fidelity | Structure | Recall | Reward | Status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0–96 | 96 | 1.0 | 1.0 | 0.8843 | 1.0 | 0.9955 | complete |
| 1 | 96–192 | 96 | 1.0 | 1.0 | 0.8603 | 0.9948 | 0.9936 | complete |
| 2 | 192–288 | 96 | 1.0 | 1.0 | 0.8521 | 1.0 | 0.9948 | complete |
| 3 | 288–384 | 96 | 1.0 | 1.0 | 0.8687 | 0.9948 | 0.9928 | complete |
| 4 | 384–480 | 96 | 1.0 | 1.0 | 0.8641 | 0.9844 | 0.9927 | complete |
| 5 | 480–576 | 96 | 1.0 | 1.0 | 0.8747 | 1.0 | 0.9935 | complete |
| 6 | 576–672 | 96 | 1.0 | 1.0 | 0.8768 | 0.9948 | 0.9926 | complete |
| 7 | 672–768 | 96 | 1.0 | 1.0 | 0.8637 | 0.9896 | 0.9934 | complete |
| 8 | 768–864 | 96 | 1.0 | 1.0 | 0.8631 | 1.0 | 0.9946 | complete |
| 9 | 864–960 | 96 | 1.0 | 1.0 | 0.8922 | 1.0 | 0.9931 | complete |

Shard 0 completes normally in about 183 seconds with zero failure, fallback,
or decode timeout. Its diagnostic AgentV bundle is 0/5 with zero execution
errors because four bounded suites and the complete RICO suite are absent.
Shard 1 completes normally in about 218 seconds with the same zero
failure/fallback/timeout counts and diagnostic AgentV status.
Shard 2 completes normally in about 199 seconds, also with zero
failure/fallback/timeout counts.
Shard 3 completes normally in about 222 seconds with zero recorded
failure/fallback/timeout counts.
Shard 4 completes normally in about 195 seconds with the same zero
failure/fallback/timeout counts.
Shard 5 completes normally in about 226 seconds with zero
failure/fallback/timeout counts.
Shard 6 completes normally in about 210 seconds with zero
failure/fallback/timeout counts.
Shard 7 completes normally in about 240 seconds with zero
failure/fallback/timeout counts.
Shard 8 completes normally in about 242 seconds with zero
failure/fallback/timeout counts.
Shard 9 completes normally in about 243 seconds with zero
failure/fallback/timeout counts.

**Current verdict:** 960/1500 rows are complete. This is a partial diagnostic,
not a full-RICO, ship-gate, champion, or promotion claim.
