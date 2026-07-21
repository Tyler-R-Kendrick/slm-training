# E697 — required parent ordering

Date: 2026-07-21
Status: completed positive tradeoff; not ship

E697 orders a planned parent before planned families reachable through its
required, non-alternative schema paths. Both independently capped Held-out
replays completed with exit 0, no timeout or fallback, and emitted AgentEvals
JSONL plus AgentV SDK bundles after all 135 compiler tests passed.

| Held-out `n=5` | E694 v150 | E697 r2 v157 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict v2 / coverage | 0.8000 / 1.0000 | 0.8000 / 1.0000 |
| fidelity / validity | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| structure / component recall | 0.6624 / 0.7933 | 0.6826 / 0.8433 |
| reward | 0.9634 | 0.9610 |
| AST node / edge F1 | 0.7754 / 0.5901 | 0.8062 / 0.5537 |
| latency p50 / p95 | 3414.50 / 5951.17 ms | 3316.05 / 6799.36 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

R1 correctly moved Form first but then treated the broad Stack collection as a
concrete owner, emitting an empty standalone Stack. Meaningful-v1 fell to 0.8
and reward to 0.7688, so r1 is implementation-confounded and excluded.
Corrected r2 defers such opaque required collections while concrete families
remain. Only Form changes: the submit Button and email Input are now owned by
Form, the standalone submit declaration disappears, and structure, recall, and
node F1 improve. `hint.title` is still duplicated in a second Button and both
Callout fields, so semantic mismatch and spam remain. Edge F1 and reward slip;
latency movement is not a performance claim.

Retain corrected v157 as a positive structural tradeoff, not ship evidence.
This is one reused scratch checkpoint and Held-out `n=5`. No checkpoint was
created, synced, or promoted.

Evidence: [JSON](iter-e697-required-parent-order-20260721.json).
