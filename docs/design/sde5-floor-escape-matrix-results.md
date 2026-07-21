# SLM-210 (SDE5-03): prompt-plan × grammar-mass × high-debt floor-escape matrix (slm210-sde5-floor-escape-matrix-20260721)

Matrix set: `sde5_floor_escape_matrix`

Version: `sde5-03-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

Combining prompt-derived plan/semantic soft features, GrammarAlignedMassPolicy / ASAp-compatible legal-mass scoring, and fixed-budget high-debt exposure produces a reproducible strict meaning-v2 > 0 signal without repeated-subtree or inventory-coverage gaming, relative to matched controls.

## Falsifier

No cell exceeds strict meaning-v2 zero with clean evidence, or an apparent gain disappears under anti-gaming, inventory-removal, retry, or seed controls.

## Cells

| cell | plan_soft | grammar_mass | asap | exposure | seed | selected | unique_groups | max_group | mean_debt | anti_gaming |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | False | False | False | uniform | 0 | 120 | 40 | 5 | 1.8039 | False |
| C1 | True | False | False | uniform | 0 | 120 | 40 | 5 | 1.8039 | True |
| C2 | False | True | True | uniform | 0 | 120 | 40 | 5 | 1.8039 | True |
| C3 | False | False | False | preregistered_composite | 0 | 120 | 37 | 5 | 2.3993 | True |
| C4 | True | True | True | uniform | 0 | 120 | 40 | 5 | 1.8039 | True |
| C5 | True | False | False | preregistered_composite | 0 | 120 | 37 | 5 | 2.3993 | True |
| C6 | False | True | True | preregistered_composite | 0 | 120 | 37 | 5 | 2.3993 | True |
| C7 | True | True | True | preregistered_composite | 0 | 120 | 37 | 5 | 2.3993 | True |
| C8 | True | True | True | debt_weight_permuted | 0 | 120 | 37 | 5 | 2.3955 | True |
| C0 | False | False | False | uniform | 1 | 120 | 39 | 5 | 1.7788 | False |
| C1 | True | False | False | uniform | 1 | 120 | 39 | 5 | 1.7788 | True |
| C2 | False | True | True | uniform | 1 | 120 | 39 | 5 | 1.7788 | True |
| C3 | False | False | False | preregistered_composite | 1 | 120 | 37 | 5 | 2.3993 | True |
| C4 | True | True | True | uniform | 1 | 120 | 39 | 5 | 1.7788 | True |
| C5 | True | False | False | preregistered_composite | 1 | 120 | 37 | 5 | 2.3993 | True |
| C6 | False | True | True | preregistered_composite | 1 | 120 | 37 | 5 | 2.3993 | True |
| C7 | True | True | True | preregistered_composite | 1 | 120 | 37 | 5 | 2.3993 | True |
| C8 | True | True | True | debt_weight_permuted | 1 | 120 | 37 | 5 | 2.3955 | True |
| C0 | False | False | False | uniform | 2 | 120 | 40 | 5 | 1.8101 | False |
| C1 | True | False | False | uniform | 2 | 120 | 40 | 5 | 1.8101 | True |
| C2 | False | True | True | uniform | 2 | 120 | 40 | 5 | 1.8101 | True |
| C3 | False | False | False | preregistered_composite | 2 | 120 | 37 | 5 | 2.3993 | True |
| C4 | True | True | True | uniform | 2 | 120 | 40 | 5 | 1.8101 | True |
| C5 | True | False | False | preregistered_composite | 2 | 120 | 37 | 5 | 2.3993 | True |
| C6 | False | True | True | preregistered_composite | 2 | 120 | 37 | 5 | 2.3993 | True |
| C7 | True | True | True | preregistered_composite | 2 | 120 | 37 | 5 | 2.3993 | True |
| C8 | True | True | True | debt_weight_permuted | 2 | 120 | 37 | 5 | 2.3955 | True |

## Disposition

**inconclusive**

Fixture wiring only: selection, exposure equality, and anti-gaming scheduling are exercised, but no model was trained or evaluated on strict meaning-v2.

## Honest caveats

- Synthetic fixture: no model, checkpoint, verifier labels, or AgentV evaluation were used.
- GrammarAlignedMassPolicy and ASAP decode are recorded as config levers and exercised on a synthetic candidate path only; live decode wiring is unchanged.
- Prompt-derived plan soft features are mocked as predicted provenance; production arms must use prompt_semantic_plan() and never gold AST/placeholder/plan fields.
- Strict meaning-v2 was not measured; this is a wiring/preregistration artifact, not a ship claim.
- Real floor-escape evaluation requires managed GPU orchestration and durable checkpoints.

## Reproducibility

```bash
python -m scripts.run_quality_matrix --matrix-set sde5-floor-escape --mode fixture
```

