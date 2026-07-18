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

**Current verdict:** 480/1500 rows are complete. This is a partial diagnostic,
not a full-RICO, ship-gate, champion, or promotion claim.
