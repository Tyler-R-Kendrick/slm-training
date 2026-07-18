# E354–E355 RICO shard and partial merge — 2026-07-17

E354 evaluates RICO rows 64–127 with the E350 policy. The 64-row CPU shard
completed in 110.7 seconds under the hard 300-second cap. Parse is 1.0,
meaningful rate 0.9063, fidelity 0.2913, structure 0.3004, component recall
0.5104, and reward 0.6755. Six examples fail meaningfulness due to low
component recall. Numeric RICO thresholds pass, but AgentV correctly remains
0/1 because the artifact is a diagnostic subset with `n=64`.

E355 merges E353 rows 0–63 and E354 rows 64–127. The aggregation completed in
1.5 seconds and validates contiguous, non-overlapping coverage under one
checkpoint hash and evaluation policy. The exact 128-row aggregate has parse
1.0, meaningful rate 0.9453, fidelity 0.2733, structure 0.2719, component
recall 0.5104, reward 0.7002, and seven low-recall failures. AgentV remains
0/1 because coverage is only 128/1500.

**Verdict:** retain both the new shard evidence and the merge path. The
campaign remains diagnostic until all 1500 rows are present.
