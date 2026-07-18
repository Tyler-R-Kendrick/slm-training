# E387–E388 structural capacity RICO 336-row evidence — 2026-07-18

E387 applies the unchanged E381 capacity-safe structural decoder to frozen
RICO rows 288–336. All 48 rows parse and are meaningful. Placeholder fidelity
is 0.9826, structural similarity 0.6278, component recall 0.9549, and reward
0.9929, with no row failures, fallbacks, or decode timeouts. The shard
completed in approximately 87 seconds under the hard 290-second process cap.

E388 canonically merges E377, E378, E381, E383, E385, and E387. It verifies
contiguous, non-overlapping rows 0–336, identical checkpoint SHA, and identical
evaluation policy. The exact 336-row aggregate has parse and meaningful rate
1.0, fidelity 0.9893, structure 0.6366, component recall 0.9764, reward 0.9949,
and zero failures. The merge completed in 1.5 seconds.

The merged AgentV bundle fails closed because the four bounded suites are not
part of this merge and RICO coverage remains 336/1500. E376 separately records
the bounded 4/4 pass.

**Verdict:** retain the unchanged capacity-safe policy and continue disjoint
48-row shards. This is strong partial evidence, not a full-RICO or production
ship result.
