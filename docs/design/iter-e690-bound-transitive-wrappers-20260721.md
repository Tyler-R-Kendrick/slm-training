# E690 — bound transitive wrappers

Date: 2026-07-21
Status: completed positive tradeoff; retained; not ship

E690 composes E689's parent-schema child disambiguation with role-bound
transitive wrapper continuation. TabItem remains eligible while either bound
tab trigger is missing, then stops serving as a generic wrapper for unrelated
roles. The independently capped full Held-out replay completed with exit 0,
no timeout or fallback, and emitted AgentEvals JSONL plus an AgentV SDK bundle.

| Held-out `n=5` | E689 v144 | E690 v145 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict v2 / coverage | 0.4000 / 1.0000 | 0.4000 / 1.0000 |
| fidelity / validity | 0.8667 / 0.9200 | 0.8333 / 0.9000 |
| structure / component recall | 0.6019 / 0.6533 | 0.6108 / 0.6533 |
| reward | 0.9210 | 0.9110 |
| AST node / edge F1 | 0.7023 / 0.6061 | 0.7312 / 0.6294 |
| latency p50 / p95 | 3457.63 / 6325.63 ms | 2836.79 / 6154.14 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

The tabs output now has exactly two TabItems, and the details pair is correctly
carried by `Callout("info", title, body)`. Heading and overview are still
inserted as raw `TabItem.content` strings rather than TextContent children,
leaving required-placeholder and schema-value-role failures. The structural,
AST, and latency gains are retained alongside the small fidelity, validity,
and reward regressions.

Retain v145 as the cleaner structural research baseline, not ship evidence.
The next lever must require a component/list carrier at TabItem.content and
place heading/overview in TextContent children. No checkpoint was created,
synced, or promoted.

Evidence: [JSON](iter-e690-bound-transitive-wrappers-20260721.json).
