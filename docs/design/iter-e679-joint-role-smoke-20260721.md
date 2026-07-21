# E679 — independent joint-role Smoke confirmation

Date: 2026-07-21
Status: completed negative confirmation; not ship

E679 replaced the invalid combined E677 attempt with an independently
terminal full Smoke replay of retained v134. The capped CPU run evaluated all
three Smoke records, completed without timeout or fallback, and emitted both
AgentEvals JSONL and an AgentV SDK bundle.

| Smoke `n=3` | E679 v134 |
| --- | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 |
| strict meaningful v2.3.0 | 0.3333 |
| fidelity / validity | 0.8056 / 0.8833 |
| structure / component recall | 0.6878 / 0.5833 |
| reward | 0.8907 |
| AST node / edge F1 | 0.8111 / 0.3556 |
| latency p50 / p95 | 1550.39 / 7455.96 ms |
| timeout / fallback | 0 / 0 |
| AgentV | 0/1 |

Only the single-button record clears strict v2. The hero omits visible
`:smoke.hero.kicker` and places `:smoke.hero.subtitle` in `CardHeader.title`;
the informational-callout record omits visible `:smoke.callout.heading`.
These failures expose a general planner limitation: E676 groups roles only
when one component covers an entire namespace, so larger namespaces are not
partitioned into jointly coverable subsets and unmatched text roles receive no
carrier.

Retain v134 as the positive OOD research baseline, but reject the hypothesis
that its current role grouping generalizes across Smoke. The next lever must
use maximal schema-valid role subsets and a schema-valid carrier for residual
visible text roles. This is a three-record diagnostic confirmation, not ship
evidence. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e679-joint-role-smoke-20260721.json).
