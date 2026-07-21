# SLM-186 (FFE0-04): Verified-utility ladder fixture (slm186-verified-utility-20260721)

Matrix set: `slm186_verified_utility`

Version: `verified-utility-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

A multi-factor verified-utility ladder can rank OpenUI candidates while keeping every factor explicit, availability-labeled, and auditable for Goodhart gaming.

## Falsifier

The scalarized ranking contradicts the lexicographic ranking on obviously dominated or abstained candidates, or small weight perturbations within the permitted ranges produce frequent rank reversals.

## Fixture candidates

| Candidate | Scenario | Scalar | Lex rank | Pareto | Notes |
| --- | --- | --- | --- | --- | --- |
| canary_wrong_binding | canary | 0.557 | 5 | False | canary, dominated_by=dominant |
| abstained | abstained | -0.058 | 7 | True | abstained, abstained, pareto_optimal |
| dominated | pareto_dominated | 0.452 | 6 | False | pareto_dominated, dominated_by=dominant,cheap_near_gold,no_judge |
| dominant | pareto_dominant | 0.842 | 1 | True | pareto_dominant, pareto_optimal |
| cheap_near_gold | economy | 0.655 | 3 | True | economy, pareto_optimal |
| canary_missing_component | canary | 0.509 | 2 | False | canary, dominated_by=dominant |
| no_judge | partial_data | 0.626 | 4 | False | partial_data, dominated_by=dominant |

## Rankings

**Scalar ranking (best first):** dominant, cheap_near_gold, no_judge, canary_wrong_binding, canary_missing_component, dominated, abstained

**Lexicographic ranking (best first):** dominant, canary_missing_component, cheap_near_gold, no_judge, canary_wrong_binding, dominated, abstained

**Pareto front:** abstained, dominant, cheap_near_gold

## Abstention economics

Accepted: 6; Abstained: 1; mean utility accepted: 0.668; value of abstention: 0.068.

## Sensitivity

Perturbations: 40; rank reversals: 6 (rate: 0.150).

## Goodhart canary summary

Canary cases wired: 18; overall canary strict rate: 1.000.

| Slice | Cases | Strict rate |
| --- | --- | --- |
| canary_ast_similar_missing_component | 3 | 1.000 |
| canary_canonical_equivalent_positive | 3 | 1.000 |
| canary_overlong_economy_violation | 3 | 1.000 |
| canary_render_semantics_mismatch | 3 | 1.000 |
| canary_right_inventory_wrong_hierarchy | 3 | 1.000 |
| canary_right_role_wrong_binding | 3 | 1.000 |

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The verified-utility ladder, scalar/lexicographic rankings, Pareto frontier, abstention economics, and sensitivity analysis are wired and exercised on deterministic synthetic candidates.  Real eval records, independent judge scores, and human pair-preference data are required before claiming any floor-escape.  The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_eval``.

## Honest caveats

- All utilities are synthetic fixtures; real eval records would change   numeric rankings and may expose rank reversals not seen here.
- Independent judge score and human pair preference are marked   ``unavailable`` or synthesized; they must not be treated as real   human evidence.
- The Goodhart canary slices are deterministic and scored by the   current ``binding_aware_meaningful_v2`` metric; new slices may be   added as additional gaming channels are identified.
- No ship-gate claim is made; this is wiring evidence only.

## Next steps

1. Replace synthetic utilities with real suite scores and judge    envelopes.
2. Calibrate the weight manifest against held-out human preferences.
3. Re-run sensitivity analysis after every metric or weight change.
4. Close the loop with the SLM-186 Goodhart canary suite: any metric    change that flips a canary case must be documented and justified.

## Reproducibility

```bash
python -m scripts.run_verified_utility_audit --mode describe
python -m scripts.run_verified_utility_audit --mode fixture
python -m scripts.run_verified_utility_audit --mode analyze-history PATH
python -m scripts.run_verified_utility_audit --mode sensitivity
```
