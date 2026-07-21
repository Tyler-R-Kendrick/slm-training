# E685 — numbered tab role carriers

Date: 2026-07-21
Status: completed negative; contract change retained; not ship

E685 normalizes trailing digits in semantic roles (`tab1`/`tab2` → `tab`),
maps that role to the public `TabItem.trigger` property, and permits a visible
role with exactly one schema-compatible family to plan that carrier. The
binding-aware metric advances to v2.6.0; ship thresholds remain disabled and
unchanged.

The first attempt (`e685-numbered-tab-carriers-r1`) wrote complete artifacts,
but its supervisor lost the terminal payload after output truncation. It is
excluded from evidence. The independently capped `r2` replay completed with
exit 0, no timeout or fallback, and emitted AgentEvals JSONL plus an AgentV SDK
bundle.

| Held-out `n=5` | E684 v138 | E685 v139 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 0.8000 | 1.0000 / 0.8000 |
| strict v2 / coverage | 0.4000 / 1.0000 | 0.4000 / 1.0000 |
| fidelity / validity | 0.7000 / 0.8200 | 0.7000 / 0.8200 |
| structure / component recall | 0.5108 / 0.5733 | 0.5108 / 0.5733 |
| reward | 0.8602 | 0.8602 |
| AST node / edge F1 | 0.6218 / 0.5203 | 0.6218 / 0.5203 |
| latency p50 / p95 | 3289.20 / 18338.00 ms | 3669.79 / 20040.68 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Every quality metric and every prediction is identical to E684. In particular,
the tabs record remains `root = TextContent(":held.tabs.tab2")`; neither
`Tabs` nor either planned `TabItem` carrier reaches the prediction. The
hypothesis is rejected: child-carrier planning cannot repair an absent parent
family.

Retain v2.6.0/v139 only as a truthful role-contract representation change, not
as quality or ship evidence. The next lever must diagnose parent `Tabs`
reachability before adding more role aliases. No checkpoint was created,
synced, or promoted.

Evidence: [JSON](iter-e685-numbered-tab-carriers-20260721.json).
