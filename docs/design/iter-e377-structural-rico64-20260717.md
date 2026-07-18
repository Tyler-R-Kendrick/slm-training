# E377 structural RICO 64-row shard — 2026-07-17

E377 extends the E375 structural terminal-coverage policy from 16 to the first
64 frozen rows of the leakage-filtered 1,500-row RICO suite. The unchanged E368
checkpoint uses visible-slot floor `-1`, component-plan weight 2,
slot-component weight 8, and a 320-token choice canvas.

All 64 rows parse and are meaningful. Placeholder fidelity is 0.9922,
structural similarity 0.6695, component recall 1.0, and reward 0.9971, with no
row-level failures or fallbacks. Latency is 2.51s p50 / 4.66s p95.

The command completed in approximately 167 seconds under the external
290-second interrupt and forced kill ten seconds later. AgentV correctly
reports 0/1 because 64/1500 rows remain partial evidence.

**Verdict:** retain the policy and continue disjoint RICO shards. This is strong
partial evidence, not a full-RICO or ship result.
