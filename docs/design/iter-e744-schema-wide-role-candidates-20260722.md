# E744 — schema-wide typed role candidates

**Date:** 2026-07-22  
**Decision:** retain treatment as the next diagnostic control; no promotion  
**Evidence:** [`iter-e744-schema-wide-role-candidates-20260722.json`](iter-e744-schema-wide-role-candidates-20260722.json)

E744 tests the existing `semantic_role_schema_candidates` lever after E743
showed that all four hero roles had zero direct candidates when role inference
was restricted to component names mentioned in the prompt. The treatment maps
only declared `RuntimeSymbol.semantic_role` values into compatible official
schema properties. It does not inspect marker spellings and adds no free-form
completion channel.

Both arms reuse the unchanged local E735 checkpoint and three frozen smoke
records. They ran locally on CPU with `strict_compiler_tree`, honest slot
contracts, semantic-plan family weights 4/2, coverage-close weight 2,
schema-role and semantic-role weights 2, component-plan and root-arity weights
1, a 160-symbol canvas, an eight-second per-record guard, and the two-minute
command cap. Their complete 235-field effective configurations differ only in
`semantic_role_schema_candidates` (`false` versus `true`).

| Arm | Plan apps / changes | Close apps / changes | Tokens | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| schema candidates off | 3 / 2 | 3 / 3 | 61 | 1.0000 | 1.0000 | 0.0000 | 0.8889 | 0.9333 | 0.6628 | 0.5000 | 0.9037 | 993 / 3573 ms | 0/1 |
| schema candidates on | 5 / 3 | 4 / 3 | 77 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.7535 | 0.7500 | 0.9370 | 1714 / 4074 ms | 0/1 |

The treatment replaces the hero's incompatible `Button` with role-compatible
`CardHeader` and `TextContent` families and restores the callout heading as a
separate `TextContent`. Fidelity rises 0.1111, validity 0.0667, structure
0.0907, recall 0.25, and reward 0.0333. The hero still repeats its body symbol
three times and strict-v2 reports semantic-role mismatch, duplicate-subtree
spam, and placeholder spam. AgentV remains 0/1; the small smoke subset is not
ship evidence and latency is not promoted.

Retain the existing treatment as the next control, but do not promote a
checkpoint or readiness claim. The next repair belongs in shared semantic-plan
role obligations: a descendant family can receive multiple role bindings while
receiving zero planned instances, leaving cardinality unconstrained.
