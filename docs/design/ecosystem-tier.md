# Ecosystem tier vs production core

**Status:** Classification registry + tier-split scoreboard implemented.
Measurement only — not a ship gate and not a claim that library growth improves
generation quality.

## Why

The OpenUI **library** (components, roles, operators) is growing. That growth is
useful for coverage and CAP/DSH work, but it is easy to conflate with the
**production core**: constrained decode, certified forest closure, exact-closure
honesty, and the formal preflights that protect those invariants.

Rule:

> Measure library impact on generation metrics and formal preflight success
> **separately** from the pure production core. Ecosystem growth never greens a
> core formal preflight, and core formal status is never a function of library
> inventory size.

## Tiers

| Tier | Owns | Examples |
| --- | --- | --- |
| `production_core` | Decode legality, bootstrap surface, core formal modules | DSH3-04 six local ops; Stack/Card/Button/… bootstrap components; `Forest` / `Trace` / `ExactClosure` / `DecodeInvariants` / `ListSet`; `parse_rate` / `meaningful_program_rate` |
| `ecosystem_library` | Growing pack inventory + library-sensitive metrics/formal | Modal/Tabs/…; topology & history ops; extra semantic roles; `StructuralMetrics` / `EcosystemTier`; `component_type_recall` / `structural_similarity` / `placeholder_fidelity` |

Authority: `src/slm_training/dsl/ecosystem_tier.py`  
Committed constants: `src/slm_training/resources/ecosystem_tier_registry.json`  
Scoreboard: `src/slm_training/evals/ecosystem_tier_scoreboard.py`  
CLI: `python -m scripts.measure_ecosystem_tier`

## Generation metrics split

| Metric family | Tier | Rationale |
| --- | --- | --- |
| `parse_rate`, `syntax_parse_rate`, `raw_syntax_validity`, `meaningful_program_rate` | production_core | Syntax / constrained legality; library-agnostic |
| `component_type_recall`, `structural_similarity`, `tree_edit_similarity`, `placeholder_fidelity`, `reward_score` | ecosystem_library | Sensitive to component/role inventory and packing density |
| Anything else | `unassigned` | Fail open for reporting; never auto-attached to core |

Ship gates in `ship_gates.py` remain the multi-suite policy. This split is a
**diagnostic scoreboard**, not a second promotion path.

## Formal preflight split

| Lean modules | Tier |
| --- | --- |
| `ListSet`, `Forest`, `Trace`, `ExactClosure`, `DecodeInvariants` | production_core |
| `StructuralMetrics`, `EcosystemTier` | ecosystem_library |

`LeverProofLean.EcosystemTier` proves the partition and that core formal success
ignores ecosystem library size (`core_success_ignores_library_size`,
`library_growth_preserves_core_success`).

## Measure

```bash
# Inventory + empty metrics (fixture/demo)
python -m scripts.measure_ecosystem_tier --fixture-demo \
  --write-registry --check-registry

# Split an eval scoreboard fragment
python -m scripts.measure_ecosystem_tier \
  --metrics path/to/metrics.json \
  --formal-statuses path/to/formal.json \
  --out-json docs/design/ecosystem-tier-scoreboard.json \
  --out-md docs/design/ecosystem-tier-scoreboard.md

# Optional: bounded lake build probe for formal module statuses
python -m scripts.measure_ecosystem_tier --probe-lean --fixture-demo
```

## Honesty

- Fixture/demo scoreboards set `honesty.ship_claim = false`.
- A high ecosystem `component_type_recall` with a failed core formal rollup is
  a **library win on a broken core** — report both; do not average them.
- A proved core formal rollup with weak ecosystem metrics is a **core-safe,
  library-incomplete** state — also report both.
- I14: rejecting a library approach never weakens core decode invariants.

## Related

- Core formal claims: [`core-formal-claims.md`](core-formal-claims.md)
- Decode invariants: [`decode-invariants.md`](decode-invariants.md)
- Core local operators: [`dsh3-04-core-local-operators-20260723.md`](dsh3-04-core-local-operators-20260723.md)
- Ops vocab (I13): `src/slm_training/dsl/ops_vocab.py`
