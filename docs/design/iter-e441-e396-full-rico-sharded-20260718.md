# E441 E396 full-RICO sharded evaluation — 2026-07-18

E441 evaluates the 1,500-row `rico_held` suite for E396 in deterministic
96-row shards so every process remains below the non-negotiable five-minute
limit. Every shard uses the matched honest policy: HF context from local files,
320-token grammar LTR, component-plan decode weight 2, slot-component decode
weight 8, honest constrained slot contract, eight generation steps, and at
most three attempts. Each command has an external 290-second interrupt and
ten-second forced kill. Only normally completed shards count.

| Shard | Rows | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward | Status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0–96 | 96 | 1.0 | 1.0 | 1.0 | 0.6568 | 0.9158 | 0.9989 | complete |
| 1 | 96–192 | 96 | 1.0 | 0.9688 | 1.0 | 0.6262 | 0.8594 | 0.9665 | complete |
| 2 | 192–288 | 96 | 1.0 | 0.9896 | 1.0 | 0.6186 | 0.8576 | 0.9867 | complete |
| 3a | 288–336 | 48 | 1.0 | 1.0 | 1.0 | 0.6343 | 0.8090 | 0.9973 | complete |
| 3b | 336–384 | 48 | 1.0 | 1.0 | 1.0 | 0.6401 | 0.8993 | 0.9991 | reused E399 |
| 4 | 384–480 | 96 | 1.0 | 0.9688 | 1.0 | 0.6440 | 0.8490 | 0.9661 | complete |
| 5 | 480–576 | 96 | 1.0 | 0.9688 | 1.0 | 0.6429 | 0.8455 | 0.9672 | complete |
| 6 | 576–672 | 96 | 1.0 | 0.9896 | 1.0 | 0.6371 | 0.8698 | 0.9870 | complete |
| 7 | 672–768 | 96 | 1.0 | 1.0 | 1.0 | 0.6432 | 0.8707 | 0.9987 | complete |
| 8 | 768–864 | 96 | 1.0 | 0.9896 | 1.0 | 0.6310 | 0.9010 | 0.9865 | complete |
| 9 | 864–960 | 96 | 1.0 | 0.9688 | 1.0 | 0.6402 | 0.8212 | 0.9668 | complete |
| 10 | 960–1056 | 96 | 1.0 | 0.9896 | 1.0 | 0.6238 | 0.8542 | 0.9879 | complete |
| 11 | 1056–1152 | 96 | 1.0 | 0.9792 | 1.0 | 0.6415 | 0.8628 | 0.9771 | complete |
| 12 | 1152–1248 | 96 | 1.0 | 0.9896 | 1.0 | 0.6482 | 0.8637 | 0.9877 | complete |
| 13 | 1248–1344 | 96 | 1.0 | 0.9896 | 1.0 | 0.6541 | 0.9028 | 0.9873 | complete |
| 14 | 1344–1440 | 96 | 1.0 | 0.9792 | 0.9896 | 0.6426 | 0.8681 | 0.9775 | complete |
| 15 | 1440–1500 | 60 | 1.0 | 0.9833 | 1.0 | 0.6351 | 0.8361 | 0.9826 | complete |

All new shards completed normally. Their diagnostic AgentV envelopes are 0/5
because four required suites are absent and RICO is a subset; all have zero
execution errors. Decoded record times for shards 0–9 are 141.3, 171.3, 159.7,
89.6, 151.1, 170.7, 153.1, 162.0, 162.1, and 159.9 seconds, excluding reused shard 3b.
Shards 1, 2, 4, 5, and 6 record three, one, three, three, and one
low-component-recall failures; shards 8 and 9 add one and three. No timed-out
process contributes evidence. Shard 10 decodes in 169.8 seconds and adds one
low-component-recall failure. Shard 11 decodes in 166.2 seconds and adds two.
Shard 12 decodes in 162.3 seconds and adds one.
Shard 13 decodes in 170.2 seconds and adds one.
Shard 14 decodes in 156.9 seconds, adds two low-recall failures, and is the
first shard below perfect placeholder fidelity.
Shard 15 decodes in 109.8 seconds and adds one low-recall failure.

Rows 336–384 reuse E399 because its checkpoint SHA and complete evaluation
policy are identical to E441. That prior run completed normally with zero
execution errors and decoded in 104.5 seconds; it is not rerun.

## Canonical full-suite aggregate

The canonical shard merger verified identical checkpoint SHA and evaluation
policy, exact contiguous `[0, 1500)` coverage, and unique record IDs. The
merged artifact is
`outputs/runs/e441-e396-full-rico-merged-r1/eval_rico_held.json`.

| n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward | Low-recall failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1500 | 1.0 | 0.9847 | 0.9993 | 0.6390 | 0.8652 | 0.9827 | 23 |

The aggregate is not a diagnostic subset. Median/p95 latency is
1584.6/3491.0 ms, aggregate decode time is 2560.4 seconds, fallback and decode
timeout counts are zero, and the full-RICO AgentV result passes 1/1 with zero
execution errors.

**Verdict:** the full-RICO evaluation is complete. Placeholder fidelity is
below 1.0 and 23 low-recall failures remain. E442 subsequently combines this
artifact with the four complete bounded suites and passes the authoritative
five-suite ship gates; production HF ship still requires durable bucket sync.
