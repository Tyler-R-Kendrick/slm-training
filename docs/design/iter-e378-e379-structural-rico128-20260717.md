# E378–E379 structural RICO 128-row evidence — 2026-07-17

E378 evaluates the unchanged E375 structural policy on the disjoint frozen
RICO shard at rows 64–128. All 64 rows parse and are meaningful; fidelity is
0.9909, structure 0.6508, component recall 0.9779, and reward 0.9951, with no
row-level failures. The command completes in approximately 212 seconds under
the hard 290-second interrupt.

E379 uses the canonical shard merger to combine E377 rows 0–64 and E378 rows
64–128. It validates contiguous, non-overlapping coverage, identical checkpoint
SHA, and identical evaluation policy. The exact 128-row aggregate has parse
and meaningful rate 1.0, fidelity 0.9915, structure 0.6602, component recall
0.9889, reward 0.9961, and no failures. The merge completes in 1.6 seconds.

The merged AgentV bundle fails closed because the four bounded suites are not
part of this merge and RICO coverage remains 128/1500. E376 separately records
the bounded 4/4 pass.

**Verdict:** retain the cumulative evidence and continue disjoint RICO shards.
No full-RICO or production ship claim is supported yet.
