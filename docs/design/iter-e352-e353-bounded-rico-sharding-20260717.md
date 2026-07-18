# E352–E353 bounded RICO sharding and fail-closed gates — 2026-07-17

E352 validates deterministic suite offsets and the corrected evidence gate on
RICO rows 16–31. The 16-row shard completed in 24.7 seconds. Its numeric
quality clears the RICO thresholds—meaningful 1.0, structure 0.2458, component
recall 0.5104, and reward 0.7402—but AgentV correctly fails 0/1 because the
artifact is both a diagnostic subset and `n=16`, below the production
requirement of 1500.

E353 starts the resumable full-RICO campaign at rows 0–63. The 64-row shard
completed in 66.2 seconds. Parse is 1.0, meaningful rate 0.9844, fidelity
0.2553, structure 0.2435, component recall 0.5104, and reward 0.7249. One
example is non-meaningful due to low component recall. AgentV again correctly
fails 0/1 on evidence completeness, not numeric quality.

Both commands used the E350 decode policy, CPU, frozen HF context, honest slot
contract, best-NLL checkpoint
`6f0ecf7cce2ebc7c61f133c13456ac91bcd4861bd3e2f4f70a3a72473c211985`,
and the hard 300-second cap.

**Verdict:** retain the sharding and fail-closed gate path. No shard is ship
evidence by itself. Only a gap-free, non-overlapping merge covering all 1500
RICO rows may clear the RICO evidence checks.
