# E695 — role/property capacity

Date: 2026-07-21
Status: completed negative; reverted; not ship

E695 tests whether limiting planned-family role reuse to distinct public string
properties prevents Form's remaining placeholder spam. The independently
capped full Held-out replay completed with exit 0, no timeout or fallback, and
emitted AgentEvals JSONL plus an AgentV SDK bundle after all 135 compiler tests
passed.

| Held-out `n=5` | E694 v150 | E695 v151 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict v2 / coverage | 0.8000 / 1.0000 | 0.8000 / 1.0000 |
| fidelity / validity | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| structure / component recall | 0.6624 / 0.7933 | 0.6624 / 0.7933 |
| reward | 0.9634 | 0.9634 |
| AST node / edge F1 | 0.7754 / 0.5901 | 0.7754 / 0.5901 |
| latency p50 / p95 | 3414.50 / 5951.17 ms | 3379.44 / 6370.84 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Only Form changes, and only by swapping `form.title` with `hint.title` between
Callout and Form. `hint.title` is still reused in Form, Button, and FormControl,
so both `placeholder_semantic_role_mismatch` and `placeholder_spam` remain.
Every aggregate quality metric is identical; the tiny latency movement is not
a performance claim.

Reject v151 and restore v150 behavior as v152. This is one reused scratch
checkpoint and Held-out `n=5`, not ship evidence. No checkpoint was created,
synced, or promoted. A future Form lever needs per-instance/property role
assignment rather than family-level capacity accounting.

Evidence: [JSON](iter-e695-role-property-capacity-20260721.json).
