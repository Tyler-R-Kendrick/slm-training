# Ecosystem tier scoreboard

Schema: `ecosystem_tier_scoreboard/v1`

## Inventory

| Surface | Core | Ecosystem |
| --- | ---: | ---: |
| Components | 7 | 7 |
| Roles | 9 | 0 |
| Operators | 6 | 14 |
| Formal modules | 5 | 2 |

## Generation metrics (split)

### Production core

- `meaningful_program_rate`: 0.7
- `parse_rate`: 0.92
- `syntax_parse_rate`: 0.92

### Ecosystem library

- `component_type_recall`: 0.45
- `placeholder_fidelity`: 0.4
- `reward_score`: 0.48
- `structural_similarity`: 0.52

## Formal preflight (split)

- **Production core rollup:** `proved` (5/5 modules)
- **Ecosystem library rollup:** `proved` (2/2 modules)

## Separation

- Core formal ok: `True`
- Ecosystem library size: `21`
- Separation holds: `True`
- Note: Core formal preflight is evaluated on CORE_FORMAL_MODULES only; ecosystem inventory size is recorded but never consulted as a core pass condition.

Honesty: not a ship claim; production-core formal preflight is independent of library growth.
