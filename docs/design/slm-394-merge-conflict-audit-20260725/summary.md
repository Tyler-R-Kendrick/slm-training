# SLM-394 effect-derived merge conflict audit

Status: fixture wiring evidence passed; not a model or ship claim.

The audit replays every admitted exact local/topology action for the fixed
OpenUI state, then classifies every unordered pair through the production
effect-derived classifier. The legacy name-based label is recorded only for
comparison; production merge code does not inspect operator identifiers.

## Exact denominator

- complete legal-set coverage: `complete`
- replayed exact actions: `66`
- unordered action pairs: `2145`
- explained old/new disagreements: `0`

## Labels

| Label | Historical name heuristic | Effect-derived |
| --- | ---: | ---: |
| `child_order` | 152 | 152 |
| `delete_modify` | 56 | 56 |
| `disjoint` | 1352 | 1352 |
| `role_cardinality` | 582 | 582 |
| `same_node_incompatible_edit` | 3 | 3 |

All exact effects replayed through the pack authority before classification.
Incomplete/partial legal-set coverage is reported as unknown, never promoted
to a merge-safety claim. The AgentV bundle proves only audit integrity.
