# E684 — singular prose for plural schema families

Date: 2026-07-21
Status: completed mixed contract correction; retained; not ship

E684 derives singular prose matchers for plural public-schema component
families. A singular compound such as “two-tab panel” now plans one `Tabs`
container rather than multiplying the container by the preceding count. The
binding-aware metric advances to v2.5.0 because prompt-component coverage now
recognizes this schema-derived form; ship thresholds remain disabled and
unchanged.

The independently capped full Held-out replay completed without timeout or
fallback and emitted AgentEvals JSONL plus an AgentV SDK bundle.

| Held-out `n=5` | E683 v137 | E684 v138 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 0.8000 | 1.0000 / 0.8000 |
| strict v2 / coverage | 0.4000 / 0.8000 | 0.4000 / 1.0000 |
| fidelity / validity | 0.7333 / 0.8400 | 0.7000 / 0.8200 |
| structure / component recall | 0.4933 / 0.5333 | 0.5108 / 0.5733 |
| reward | 0.8702 | 0.8602 |
| AST node / edge F1 | 0.5773 / 0.5203 | 0.6218 / 0.5203 |
| latency p50 / p95 | 3344.37 / 20817.96 ms | 3289.20 / 18338.00 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

The contract correction succeeds: the tabs record now has known coverage and
explicitly reports missing `Tabs` plus missing placeholders instead of
`prompt_contract_unknown`. The quality hypothesis fails because decode emits
only `TextContent(tab2)`, not Tabs. Structure, recall, node F1, and latency
improve, while fidelity and reward regress. The other four predictions are
byte-identical to E683.

Retain metric v2.5.0 and v138 as an honest contract correction, not a quality
or ship claim. The next lever must enforce the already recognized authored
Tabs family. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e684-singular-plural-family-20260721.json).
