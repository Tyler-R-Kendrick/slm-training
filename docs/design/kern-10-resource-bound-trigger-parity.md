# KERN-10 — Python/Lean resource-bound and trigger parity (SLM-538)

Frozen cross-language fixtures prove that Python and Lean agree on:

- domain sizes / finite-search bounds (KERN-03)
- black-box unsupported lower bounds (KERN-04)
- closure-height / stabilization bounds (KERN-05)
- machine-cost totals (KERN-06)
- singleton zero-neural work (KERN-07)
- mechanism trigger / no-effect / dominance (KERN-08)
- practical computability labels (KERN-12 vocabulary; EVID-03 / KERN-10 consume it)

## Artifacts

| Path | Role |
| --- | --- |
| `src/slm_training/resources/formal/resource_bound_trigger_parity.schema.json` | Canonical fixture schema |
| `src/slm_training/resources/formal/resource_bound_trigger_parity_fixtures.v1.json` | Frozen cases + exact expect |
| `src/slm_training/formal/resource_bound_parity.py` | Generate/consume + diagnostic compare |
| `src/leverproof_lean/LeverProofLean/ResourceBoundParity.lean` | Lean mirrors (`#guard` / theorems) |
| `tests/test_formal/test_resource_bound_parity.py` | CI parity + mutation detection |
| `scripts/verify_resource_bound_parity.py` | Standalone CI entrypoint |

## Contract

- Schema id: `resource_bound_trigger_parity/v1`
- Exact integers / rational strings only; `wall_clock_claim` is always false
- Supported cases: bit-for-bit / canonically equivalent `expect` vs live owners
- Unsupported probes (`stale_schema_version`, `unknown_trigger`, `incomplete_domain`,
  `cost_model_mismatch`, `unknown_computability_label`, `duplicate_live`) fail closed
- Parity failures name `case`, `field`, `lean_module`, `lean_theorem`, `model_id`

## Boundary coverage

Zero / one / many domains, duplicates, huge products (`u64_overflow`), incomplete
domains, unknown triggers, timeout / invalid judgments, cost-model mismatch, and
stale schema versions are all present as named cases.

## Run

```bash
PYTHONPATH=src uv run pytest tests/test_formal/test_resource_bound_parity.py -q
PYTHONPATH=src uv run python -m scripts.verify_resource_bound_parity
(cd src/leverproof_lean && lake build && make proofs)
```
