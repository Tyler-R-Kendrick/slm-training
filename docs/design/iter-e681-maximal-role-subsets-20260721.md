# E681 — maximal semantic-role subsets

Date: 2026-07-21
Status: completed positive; retained; not ship

E681 partitions each visible placeholder namespace into the largest disjoint
role groups that one public-schema component can cover through distinct direct
string properties. When overlapping groups have equal cardinality, the
planner prefers the group with the most specific role candidates. On the
Smoke hero this selects `CardHeader(title, subtitle)` rather than the broader
`Callout(title, body)` pair.

The independently capped full Smoke replay completed without timeout or
fallback and emitted AgentEvals JSONL plus an AgentV SDK bundle.

| Smoke `n=3` | E680 v135 | E681 v136 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict meaningful v2.4.0 | 0.6667 | 0.6667 |
| fidelity / validity | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| structure / component recall | 0.7276 / 0.6667 | 0.8156 / 0.7500 |
| reward | 0.9530 | 0.9570 |
| AST node / edge F1 | 0.8051 / 0.4545 | 0.8778 / 0.5333 |
| latency p50 / p95 | 1734.90 / 6313.97 ms | 1637.71 / 5389.64 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

The hero no longer uses the schema-invalid `TabItem.content` path. It now
emits `CardHeader`, but positional component arguments do not yet expose their
public-schema property names to role-aware slot selection. The result repeats
and swaps title/subtitle/body placeholders across two CardHeaders, leaving one
strict role mismatch. The other two predictions are unchanged from E680.

Retain v136 for its structural, recall, reward, AST F1, and latency gains. The
unchanged strict rate prevents a stronger claim; the next lever must make
positional role selection property-aware. This remains a three-record scratch
evaluation, not ship evidence. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e681-maximal-role-subsets-20260721.json).
