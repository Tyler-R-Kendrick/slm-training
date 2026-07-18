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

Both shards completed normally. Their diagnostic AgentV envelopes are 0/5
because four required suites are absent and RICO is a subset; both have zero
execution errors. Decoded record time is 141.3 and 171.3 seconds,
respectively. Shard 1 records three low-component-recall failures. No timed-out
process contributes evidence.

**Interim status:** 192/1500 RICO rows complete. This is partial coverage, not
a ship gate, promotion, or full-RICO claim.
