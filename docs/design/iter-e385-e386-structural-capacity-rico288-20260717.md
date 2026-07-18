# E385–E386 structural capacity RICO 288-row evidence — 2026-07-17

E385 applies the unchanged E381 capacity-safe structural decoder to frozen
RICO rows 240–288. All 48 rows parse and are meaningful. Placeholder fidelity
is 0.9913, structural similarity 0.6170, component recall 0.9722, and reward
0.9954, with no row failures, fallbacks, or decode timeouts. The shard
completed in approximately 91 seconds under the hard 290-second process cap.

E386 canonically merges E377, E378, E381, E383, and E385. It verifies
contiguous, non-overlapping rows 0–288, identical checkpoint SHA, and identical
evaluation policy. The exact 288-row aggregate has parse and meaningful rate
1.0, fidelity 0.9905, structure 0.6380, component recall 0.9800, reward 0.9953,
and zero failures. The merge completed in 1.5 seconds.

The merged AgentV bundle fails closed because the four bounded suites are not
part of this merge and RICO coverage remains 288/1500. E376 separately records
the bounded 4/4 pass.

**Verdict:** retain the unchanged capacity-safe policy and continue disjoint
48-row shards. This is strong partial evidence, not a full-RICO or production
ship result.
