# E747-E751 — symbol-only RICO ownership repair

**Date:** 2026-07-22  
**Decision:** retain test-data v4 and model v212; no checkpoint promotion  
**Evidence:** [`iter-e747-e751-symbol-only-rico-ownership-20260722.json`](iter-e747-e751-symbol-only-rico-ownership-20260722.json)

The first E747 attempt correctly failed before valid evidence because the old
eval snapshot still contained the free-form target literal `contact`. Test-data
v4 now applies the symbol-only assertion after normalization and fails before
creating its output root when sanitization is disabled. Its leakage loader also
resolves relative record paths from the owning manifest checkout, so isolated
worktrees do not need copied mutable datasets.

The replacement immutable snapshot contains 51 records across smoke, held-out,
adversarial, OOD, and RICO. All 51 targets were normalized; the independent
audit found zero output-contract violations, zero sanitization fallbacks, and
zero errors. Its fingerprint is `c68deaacd88f0981e4ad98424f32811b1ac28d1472763e549396a8cc3892fd37`.

The valid five-suite E747 diagnostic identified RICO as weakest: parse 1.0,
strict-v2 0, fidelity 0.2323, structure 0.0946, and component recall 0. The
model always emitted a `Callout`, although every prompt visibly requested
repeated `Card` components. E748 proved that raising the existing semantic-plan
weight was inert. E749 made the lexer container path consume the plan score,
but E750 showed `Card` remained tied with `Callout`: the obligation builder had
invented a direct Callout action even though planned Card descendants could
own the title/body roles.

Model v212 fixes that canonical obligation calculation. Planned families now
satisfy joint roles through the public schema's reachable descendants. The
matched E751 replay changes no checkpoint, data, or lever values.

| Arm | Model | Parse | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E748 weight-only | v210 | 1.0000 | 0.0000 | 0.2323 | 0.5394 | 0.0946 | 0.0000 | 0.7067 | 0/1 |
| E749 component path, weight 4 | v211 | 1.0000 | 0.0000 | 0.2323 | 0.5394 | 0.0946 | 0.0000 | 0.7067 | 0/1 |
| E750 component path, weight 6 | v211 | 1.0000 | 0.0000 | 0.2323 | 0.5394 | 0.0946 | 0.0000 | 0.7067 | 0/1 |
| E751 reachable ownership | v212 | 1.0000 | 0.0000 | 0.6970 | 0.8182 | 0.7216 | 1.0000 | 0.8461 | 0/1 |

E751 is a real quality improvement but not a ship result. The output nests six
Cards instead of making five siblings, omits some required markers, strict-v2
remains zero, and AgentV remains 0/1. The next repair is in the canonical
semantic-plan parser/topology: exclude schema-role annotations from component
cardinality and preserve repeated families as siblings. No checkpoint was
created or synced. Every prediction contains only grammar/AST tokens, schema
enum literals, and declared template markers.
