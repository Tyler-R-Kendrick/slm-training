# DSH3-29 cost-aware ternary ECOC abstention under controlled PARTIAL coverage (SLM-404)

Status: fixture-scale synthetic demonstration; not a ship claim

## Aggregated per-level risk (summed over 4 synthetic states)

| Budget | Coverage | Arm | Detected errors | Retained wrong | Cost-weighted risk | Plain risk |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 1.00 | `cost_aware` | 3 | 1 | 1.000 | 0.250 |
| 3 | 0.75 | `cost_aware` | 0 | 3 | 3.000 | 0.750 |
| 2 | 0.50 | `cost_aware` | 0 | 2 | 0.000 | 0.500 |
| 1 | 0.25 | `cost_aware` | 0 | 0 | 0.000 | 0.000 |
| 4 | 1.00 | `uniform` | 2 | 2 | 4.000 | 0.500 |
| 3 | 0.75 | `uniform` | 0 | 3 | 3.000 | 0.750 |
| 2 | 0.50 | `uniform` | 0 | 2 | 0.000 | 0.500 |
| 1 | 0.25 | `uniform` | 0 | 0 | 0.000 | 0.000 |

## LocalFlatHead baseline (coverage structure only)

LocalFlatHead has no ECOC codeword geometry, so no cost-weighted confusion risk is computed for it; this is coverage-structure context only (gold-witnessed / forced-singleton counts), not a risk comparison.

| Budget | States | Gold witnessed | Forced singleton |
| ---: | ---: | ---: | ---: |
| 4 | 4 | 4 | 0 |
| 3 | 4 | 3 | 0 |
| 2 | 4 | 2 | 0 |
| 1 | 4 | 1 | 4 |

## Honesty

Overall cost-weighted risk: cost-aware=1.0000, uniform=1.7500. This is a hand-built, small synthetic fixture (one catastrophic action pair rotated across 4 gold labels), not a real operator_policy_corpus sample -- it demonstrates the wiring end to end, not a production readiness claim.

Decision: `fixture-scale-positive-signal` -- cost-aware ECOC codeword assignment strictly reduced cost-weighted retained-wrong-action risk versus uniform assignment on this small synthetic fixture; this is fixture-scale positive signal, not a ship claim -- real-corpus PARTIAL-coverage incidence remains too low per DSH3-28 to run this at scale, so re-run once operator_policy_corpus supplies natural PARTIAL rows before treating this as more than wiring evidence
