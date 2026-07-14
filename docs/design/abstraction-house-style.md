# Abstraction ladder and house-style policy

SLM-10 wires deterministic corpus policy; it does not report a model-quality or
ship result.

## L0-L5 contract

| Level | Input abstraction | Target determinacy | Source family |
| --- | --- | --- | --- |
| L0 | Existing DSL / AST | exact | `frontier_semantic` |
| L1 | Semantic graph | structural | `frontier_semantic` |
| L2 | Detailed specification | structural | `frontier_semantic` |
| L3 | Product requirements | house style | `frontier_product` |
| L4 | User story | house style | `frontier_user` |
| L5 | Vague request | house style | `frontier_simplified` |

Every ladder row carries required, optional, and forbidden facts; unspecified
dimensions; measured constraint coverage; and target determinacy. Frozen
artifact aliases (`semantic`, `product`, `user`, `simplified`) map to canonical
levels without a model call.

## Resolution and rejection

L0-L2 accept one exact or structural target. L3-L5 canonicalize and rank valid
candidates by immutable design-system defaults, then use canonical text as the
final tie-break. This yields one reproducible target for the same prompt and
candidate set.

| Policy axis | Default |
| --- | --- |
| Layout | column |
| Component preference | Stack, Card, TextContent, Button |
| Spacing | `m` |
| Responsive | stack on narrow viewports |
| Loading | skeleton preserves layout |
| Error | inline and recoverable |
| Content | placeholders only |

The grounding checker rejects required-fact omissions, forbidden-fact
inventions, target contradictions, invalid targets, and prose that exposes DSL
or placeholder syntax. Counterfactual pairs must substitute exactly one required
fact. Novelty selection records candidate, accepted, dropped, and near-duplicate
counts; repeated novelty signatures are capped. Modality order/presence is
rejected from materialized variants because it belongs to online augmentation.
