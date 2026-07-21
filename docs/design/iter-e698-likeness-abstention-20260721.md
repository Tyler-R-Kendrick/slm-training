# E698 — likeness abstention

Date: 2026-07-21
Status: completed positive tradeoff; not ship

E698 treats `component-like` prose as descriptive instead of an explicit
component requirement, then uses a uniquely matching public enum value to
disambiguate an otherwise ambiguous visible-role carrier. Both capped
Held-out replays completed with exit 0, no timeout or fallback, and emitted
AgentEvals JSONL plus AgentV bundles after the semantic/eval suites passed.

| Held-out `n=5` | E697 v157 | E698 r2 v159 |
| --- | ---: | ---: |
| syntax / meaningful v1 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| strict v2 / coverage | 0.8000 / 1.0000 | 0.8000 / 1.0000 |
| fidelity / validity | 1.0000 / 1.0000 | 0.9600 / 0.9760 |
| structure / component recall | 0.6826 / 0.8433 | 0.7724 / 0.8433 |
| reward | 0.9610 | 0.9514 |
| AST node / edge F1 | 0.8062 / 0.5537 | 0.8609 / 0.6888 |
| latency p50 / p95 | 3316.05 / 6799.36 ms | 3950.94 / 4602.84 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Only Form changes. The adjective `Form-like` no longer forces a public Form
whose required FormControl label has no compatible visible slot. R1 produced
Button plus Callout but omitted email; corrected r2 uses the unique `email`
enum evidence to add Input. The resulting Button/Callout/Input/Stack structure
is much closer to gold. `hint.title` remains missing, so strict stays 4/5 and
fidelity/reward slip. E699 r3 later confirms byte-identical predictions under
the restored v163 stack and metric 2.9.0.

Retain v159 behavior (restored as v163) as a positive structural tradeoff, not
ship evidence. One reused scratch checkpoint, Held-out `n=5`; no checkpoint
was created, synced, or promoted.

Evidence: [JSON](iter-e698-likeness-abstention-20260721.json).
