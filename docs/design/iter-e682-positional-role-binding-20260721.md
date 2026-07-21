# E682 — property-aware positional role binding

Date: 2026-07-21
Status: completed positive; retained; not ship

E682 exposes the active public-schema property name to role-aware slot
selection for explicitly planned positional bindings. A role-bound direct
string property may receive a compatible visible slot even when the legacy
schema placeholder annotation is absent. Unplanned component slots retain the
existing broader behavior.

The independently capped full Smoke replay completed without timeout or
fallback and emitted AgentEvals JSONL plus an AgentV SDK bundle.

| Smoke `n=3` | E681 v136 | E682 v137 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict meaningful v2.4.0 | 0.6667 | 1.0000 |
| fidelity / validity | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| structure / component recall | 0.8156 / 0.7500 | 0.8308 / 0.7500 |
| reward | 0.9570 | 0.9570 |
| AST node / edge F1 | 0.8778 / 0.5333 | 0.9030 / 0.4815 |
| latency p50 / p95 | 1637.71 / 5389.64 ms | 1439.60 / 4786.91 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

The hero now emits `CardHeader(title, subtitle)` followed by a Card containing
the kicker and body TextContent leaves. All three Smoke records clear strict
v2.4.0. Structure, node F1, and latency improve, while edge F1 regresses from
0.5333 to 0.4815; that tradeoff is retained rather than hidden.

Retain v137 as positive research evidence. This is still only a three-record
Smoke evaluation, AgentV remains 0/1, and no multi-suite ship gates were run.
The next step is an independently terminal confirmation on another suite.
No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e682-positional-role-binding-20260721.json).
