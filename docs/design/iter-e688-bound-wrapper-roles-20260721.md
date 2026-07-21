# E688 — bound wrapper roles

Date: 2026-07-21
Status: completed negative; rejected; not ship

E688 restricts transitive schema-wrapper continuation when a component has
explicit semantic-plan role bindings. The independently capped full Held-out
replay completed with exit 0, no timeout or fallback, and emitted AgentEvals
JSONL plus an AgentV SDK bundle.

The result is prediction-, metric-, and tabs-trace-identical to E687. Strict
v2.6.0 remains 2/5, structure 0.5108, component recall 0.5733, reward 0.8602,
and AgentV 0/1. `TabItem` still repeats through position 148.

The guard never activates. With the full public component inventory, `tab1`
and `tab2` each have two direct candidates: `AccordionItem` and `TabItem`.
The planner therefore binds only Callout to the details pair and TextContent
to heading/overview; there is no explicit TabItem binding to enforce. The
earlier focused unit test used a reduced component inventory and hid this
ambiguity.

Reject and revert v142. The next lever must disambiguate the two child carriers
using descendants of the already planned parent schema (`Tabs` admits
`TabItem`, not `AccordionItem`), then bind and count that unique child. No
checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e688-bound-wrapper-roles-20260721.json).
