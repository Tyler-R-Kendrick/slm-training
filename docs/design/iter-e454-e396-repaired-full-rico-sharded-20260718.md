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
| 10 | 960–1056 | 96 | 1.0 | 1.0 | 0.8529 | 0.9896 | 0.9931 | complete |
| 11 | 1056–1152 | 96 | 1.0 | 1.0 | 0.8625 | 0.9948 | 0.9944 | complete |
| 12 | 1152–1248 | 96 | 1.0 | 1.0 | 0.8642 | 0.9948 | 0.9949 | complete |
| 13 | 1248–1344 | 96 | 1.0 | 1.0 | 0.8880 | 1.0 | 0.9945 | complete |
| 14 | 1344–1440 | 96 | 1.0 | 1.0 | 0.8609 | 1.0 | 0.9956 | reused E453 |
| 15 | 1440–1500 | 60 | 1.0 | 1.0 | 0.8610 | 1.0 | 0.9957 | complete |

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
Shard 10 completes normally in about 207 seconds with zero
failure/fallback/timeout counts.
Shard 11 completes normally in about 203 seconds with zero
failure/fallback/timeout counts.
Shard 12 completes normally in about 226 seconds with zero
failure/fallback/timeout counts.
Shard 13 completes normally in about 218 seconds with zero
failure/fallback/timeout counts. Protocol-identical E453 is reused for shard
14; it completed normally in about 218 seconds and is already documented
independently. Shard 15 completes normally in about 137 seconds with zero
failure/fallback/timeout counts.

## Canonical full-suite aggregate

The canonical shard merger verified identical checkpoint SHA and evaluation
policy, exact contiguous `[0, 1500)` coverage, and unique record IDs. The
merged artifact is
`outputs/runs/e454-e396-repaired-full-rico-merged-r1/eval_rico_held.json`.

| n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward | Failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1500 | 1.0 | 1.0 | 1.0 | 0.8683 | 0.9960 | 0.9940 | 0 |

The aggregate is not a diagnostic subset. Median/p95 latency is
1807.1/5503.9 ms, fallback and decode timeout counts are zero, and the
full-RICO AgentV result passes 1/1 with zero execution errors.

**Verdict:** the repaired full-RICO evaluation is complete and materially
improves E441's meaningful rate (0.9847 → 1.0), structure
(0.6390 → 0.8683), type recall (0.8652 → 0.9960), and reward
(0.9827 → 0.9940), while restoring fidelity from 0.9993 to 1.0. This is
full-RICO evidence only, not a five-suite ship-gate, champion, promotion, or
production HF claim.
