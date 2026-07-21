# E680 — display-role aliases

Date: 2026-07-21
Status: completed positive; retained; not ship

E680 maps the general display roles `kicker` and `heading` to the public
`TextContent.text` property. Because this changes binding-aware role
correctness, the metric advances from v2.3.0 to v2.4.0; its ungated ship-policy
descriptor is aligned without changing any threshold.

The independently capped full Smoke replay completed without timeout or
fallback and emitted AgentEvals JSONL plus an AgentV SDK bundle.

| Smoke `n=3` | E679 v134 | E680 v135 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict meaningful v2 | 0.3333 | 0.6667 |
| fidelity / validity | 0.8056 / 0.8833 | 1.0000 / 1.0000 |
| structure / component recall | 0.6878 / 0.5833 | 0.7276 / 0.6667 |
| reward | 0.8907 | 0.9530 |
| AST node / edge F1 | 0.8111 / 0.3556 | 0.8051 / 0.4545 |
| latency p50 / p95 | 1550.39 / 7455.96 ms | 1734.90 / 6313.97 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

The callout record now exactly uses a `TextContent` heading followed by the
title/body `Callout` and clears strict v2.4.0. The hero covers every visible
placeholder, but still routes `subtitle` through `TabItem.content` and
duplicates `title`, so it remains the sole strict failure. This isolates the
next general limitation: E676 requires one component to cover an entire role
namespace instead of partitioning a larger namespace into maximal jointly
coverable subsets such as `CardHeader(title, subtitle)`.

Retain v135 and metric v2.4.0 as a positive research baseline. The strict
comparison crosses a metric version, but fidelity, structure, recall, reward,
edge F1, and the exact callout prediction provide independent positive
evidence. This remains a three-record scratch evaluation, not ship evidence.
No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e680-display-role-aliases-20260721.json).
