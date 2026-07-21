# E696 — scalar role literal fallback

Date: 2026-07-21
Status: completed negative; reverted; not ship

E696 tests whether a legal literal can replace a repeated visible placeholder
when a scalar string property has no unused role-compatible slot. Both
independently capped Held-out replays completed with exit 0, no timeout or
fallback, and emitted AgentEvals JSONL plus AgentV SDK bundles after all 135
compiler tests passed.

| Held-out `n=5` | E694 v150 | E696 r2 v154 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict v2 / coverage | 0.8000 / 1.0000 | 0.8000 / 1.0000 |
| fidelity / validity | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| structure / component recall | 0.6624 / 0.7933 | 0.6624 / 0.7933 |
| reward | 0.9634 | 0.9634 |
| AST node / edge F1 | 0.7754 / 0.5901 | 0.7754 / 0.5901 |
| latency p50 / p95 | 3414.50 / 5951.17 ms | 3486.32 / 7088.26 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

R1 targeted a fixed empty literal that the live constrained state did not
offer, making all five predictions byte-identical to E694; it is excluded as a
no-op implementation miss. Corrected r2 uses the live framed string route, but
Form remains byte-identical and still repeats `hint.title` three times. The only
changes are regressions: the stable empty Input name and SwitchGroup name both
become the arbitrary literal `itet`. Every aggregate quality metric remains
identical; latency movement is not a performance claim.

Reject v153/v154 and restore v152 behavior as v155. The failure confirms that
literal routing is not instance ownership; the next lever must track which
component instance owns each planned role. This is one reused scratch
checkpoint and Held-out `n=5`, not ship evidence. No checkpoint was created,
synced, or promoted.

Evidence: [JSON](iter-e696-role-literal-fallback-20260721.json).
