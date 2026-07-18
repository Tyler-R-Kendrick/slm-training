# E372–E375 structural terminal coverage — 2026-07-17

E372 tests the visible-contract-derived content floor (`-1`) after the E371
reference-kind fix. It emits more content but leaves most definitions
unreachable from the root, so canonicalization prunes them: fidelity remains
0.1257 and structure regresses to 0.1230. This falsifies raw component count as
the remaining coverage bottleneck.

E373 adds a graph-semantic invariant for marker-free structural choice streams:
the final root Stack must reference every top-level element not already
consumed by an intermediary container. On the same frozen first 16 RICO rows,
fidelity rises to 0.9271, structure to 0.6224, component recall to 0.9375, and
reward to 0.9340. One row still finalizes to an empty prediction.

The failing row contains a two-slot `StepsItem`. The content floor counts
components instead of consumed visible slots, so it requests one extra
string-bearing component after all ten slots are exhausted. E374 prevents new
slot-bearing components after slot exhaustion, but the component-count floor
still makes the same request and exactly reproduces E373.

E375 changes the structural floor's unit to distinct emitted slot tokens. The
same 16 rows then score parse 1.0, meaningful 1.0, fidelity 0.9896, structure
0.6619, component recall 1.0, and reward 0.9965 with no row-level failures.
Latency is 1.54s p50 / 5.30s p95. AgentV correctly reports 0/1 because this is
only 16/1500 RICO rows; the numeric thresholds are not full-suite evidence.

All four commands used an external 290-second interrupt and a forced kill ten
seconds later. No checkpoint was created.

**Verdict:** retain structural terminal coverage and slot-unit accounting.
E375 is a strong diagnostic improvement over E371, but neither the E368
checkpoint nor this decode policy is promoted until broader bounded-suite and
full-RICO evidence exists.
