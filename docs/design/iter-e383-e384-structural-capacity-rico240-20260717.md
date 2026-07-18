# E383–E384 structural capacity RICO 240-row evidence — 2026-07-17

E383 applies the unchanged E381 capacity-safe structural decoder to frozen
RICO rows 192–240. All 48 rows parse and are meaningful. Placeholder fidelity
is 0.9878, structural similarity 0.6253, component recall 0.9896, and reward
0.9939, with no row failures, fallbacks, or decode timeouts. The shard
completed in approximately 83 seconds under the hard 290-second process cap.

E384 canonically merges E377, E378, E381, and E383. It verifies contiguous,
non-overlapping rows 0–240, identical checkpoint SHA, and identical evaluation
policy. The exact 240-row aggregate has parse and meaningful rate 1.0,
fidelity 0.9903, structure 0.6422, component recall 0.9816, reward 0.9952, and
zero failures. The merge completed in 1.4 seconds.

The merged AgentV bundle fails closed because the four bounded suites are not
part of this merge and RICO coverage remains 240/1500. E376 separately records
the bounded 4/4 pass.

**Verdict:** retain the unchanged capacity-safe policy and continue disjoint
48-row shards. This is strong partial evidence, not a full-RICO or production
ship result.
