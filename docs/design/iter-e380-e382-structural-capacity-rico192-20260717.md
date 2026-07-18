# E380–E382 structural capacity and RICO 192-row evidence — 2026-07-17

E380 evaluated the E375 structural decoder on frozen RICO rows 128–192. One
row failed because the decoder could select a component whose content arity
exceeded the remaining visible-slot capacity. The 64-row diagnostic therefore
reached parse/meaningful 0.9844, fidelity 0.9740, structure 0.6143, component
recall 0.9531, and reward 0.9791. This negative result identified a generalized
decoder invariant rather than a row-specific exception.

The E381 decoder filters content-bearing components whose required slot count
exceeds the remaining visible slots. A focused regression test covers the
one-slot/three-slot-component case. Re-evaluating the identical 64 rows restores
parse and meaningful rate to 1.0, fidelity to 0.9896, structure to 0.6191,
component recall to 0.9609, reward to 0.9945, and zero row failures.

E382 canonically merges E377, E378, and E381. The exact contiguous 192-row
aggregate has parse and meaningful rate 1.0, fidelity 0.9909, structure 0.6465,
component recall 0.9796, reward 0.9956, and zero failures. The shard evaluation
completed under the hard 290-second process cap; the merge completed in 1.5
seconds. Future shards are limited to 48 rows for additional wall-time margin.

The merged AgentV bundle fails closed because the four bounded suites are not
part of this merge and RICO coverage remains 192/1500. E376 separately records
the bounded 4/4 pass.

**Verdict:** retain the capacity invariant and cumulative evidence; continue
smaller disjoint shards. No full-RICO or production ship claim is supported.
