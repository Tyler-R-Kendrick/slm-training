# E689 — planned child carriers

Date: 2026-07-21
Status: completed positive; retained; not ship

E689 disambiguates ambiguous visible-role carriers through cycle-safe public
schema descendants of already planned parent families. For numbered tab roles,
`Tabs` admits `TabItem` but not `AccordionItem`, so the plan can bind and count
two TabItems without a fixture-specific name rule.

The first attempt (`r1`) accidentally omitted the established positional
schema-role binding argument at an adjacent call site. Although terminal, its
unrelated form/dual-card reassignment makes it confounded and excluded from the
lever decision. Corrected `r2` completed independently with exit 0, no timeout
or fallback, and emitted AgentEvals JSONL plus an AgentV SDK bundle. The commit
hook also completed 800 model-build/model/versioning tests in 128.28 seconds,
inside the 170-second supervisor.

| Held-out `n=5` | E687 v141 | E689 r2 v144 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 0.8000 | 1.0000 / 1.0000 |
| strict v2 / coverage | 0.4000 / 1.0000 | 0.4000 / 1.0000 |
| fidelity / validity | 0.7000 / 0.8200 | 0.8667 / 0.9200 |
| structure / component recall | 0.5108 / 0.5733 | 0.6019 / 0.6533 |
| reward | 0.8602 | 0.9210 |
| AST node / edge F1 | 0.6218 / 0.5203 | 0.7023 / 0.6061 |
| latency p50 / p95 | 3037.38 / 19753.82 ms | 3457.63 / 6325.63 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

The tabs record now emits a finite Stack containing Tabs and Callout rather
than `TextContent(tab2)`, eliminating the token-cap loop. It still produces
five TabItems instead of two and places raw placeholders in `TabItem.content`,
so strict meaning does not improve. The previously strict dual-card and input
records remain strict-valid.

Retain v144 as a positive Held-out research baseline, not ship evidence. The
next lever must cap transitive TabItem continuation at planned cardinality and
route remaining roles through the planned content carriers. No checkpoint was
created, synced, or promoted.

Evidence: [JSON](iter-e689-planned-child-carriers-20260721.json).
