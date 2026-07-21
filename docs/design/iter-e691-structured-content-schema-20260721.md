# E691 — structured content schema

Date: 2026-07-21
Status: completed positive; retained; not ship

E691 fixes an overloaded schema marker at the shared choice-grammar boundary.
Content properties remain marked as slot-bearing for semantic planning, but an
array or object carrying that marker can no longer admit a raw placeholder or
use one as its minimal completion. A focused regression test covers the real
`TabItem(value, trigger, content)` contract; 155 choice/compiler tests passed.

The independently capped full Held-out replay completed with exit 0, no timeout
or fallback, and emitted AgentEvals JSONL plus an AgentV SDK bundle.

| Held-out `n=5` | E690 v145 | E691 v146 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict v2 / coverage | 0.4000 / 1.0000 | 0.4000 / 1.0000 |
| fidelity / validity | 0.8333 / 0.9000 | 0.8667 / 0.9200 |
| structure / component recall | 0.6108 / 0.6533 | 0.6658 / 0.6933 |
| reward | 0.9110 | 0.9210 |
| AST node / edge F1 | 0.7312 / 0.6294 | 0.7640 / 0.6434 |
| latency p50 / p95 | 2836.79 / 6154.14 ms | 3714.48 / 6075.57 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Only `held_out_tabs_01` changes. Its two tab bodies become
`[TextContent(heading)]` and `[TextContent(overview)]`; the other four
predictions are byte-identical. Official parse, inventory coverage, binding,
and anti-gaming checks pass. Strict v2 remains 2/5 because the changed decode
chooses `Callout("column", title, body)`, leaving only
`schema_value_role_mismatch:Callout.variant` on the tabs record.

Retain v146 as a schema-correctness fix and positive quality lever. This is one
reused scratch checkpoint and Held-out `n=5`, not a powered result or ship
claim. No checkpoint was created, synced, or promoted. The next lever should
repair enum-valued literal selection without disturbing the corrected tab tree.

Evidence: [JSON](iter-e691-structured-content-schema-20260721.json).
