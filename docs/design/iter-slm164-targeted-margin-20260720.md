# SLM-164 (SDE1-02): Confusion-targeted legal-sibling contrast margin fixture (slm164-targeted-margin-20260720)

Matrix set: `slm164_targeted_margin`

Version: `sde1-02-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no production TwoTower wiring was touched, and no ship-gate claim is made.

## Hypothesis

Targeting margin loss at confusion-prone legal siblings (e.g. Stack vs Card, same-type/different-role, rare-component substitutions) improves the separation between the expected action and its legal contrast set compared to a uniform or no-margin baseline.

## Falsifier

Targeted margin arms do not reduce violation rate or mean margin loss over the shuffled control arm, or the none baseline performs as well as the targeted arms.

## Arms

| Arm | Source | Promotable | Description |
| --- | --- | --- | --- |
| A_none | none | False | No margin loss; baseline zero metrics. |
| B_uniform | uniform | False | E228-style hardest-contrast margin over the contrast set. |
| C_targeted_hardest | targeted_hardest | False | Hardest-contrast margin focused on the targeted siblings. |
| D_targeted_weighted | targeted_weighted | False | Weighted log-sum-exp margin over all targeted contrasts. |
| E_shuffled | shuffled | False | Control arm: same rows with family labels randomly reassigned. |

## Contrast manifest

Manifest id: `slm164_synthetic_seed0`  
Families: empty_vs_child, stack_vs_card, rare_component_substitution, binder_arity, slot_pointer, same_type_different_role  
Rows: 24

## Results

| Arm | Seed | Active contrasts | Violation rate | Mean margin loss | Family coverage |
| --- | --- | --- | --- | --- | --- |
| A_none | 0 | 24 | 0.000 | 0.000 | 1.000 |
| B_uniform | 0 | 24 | 0.667 | 1.770 | 1.000 |
| C_targeted_hardest | 0 | 24 | 0.667 | 1.770 | 1.000 |
| D_targeted_weighted | 0 | 24 | 0.667 | 2.209 | 1.000 |
| E_shuffled | 0 | 24 | 0.667 | 2.209 | 1.000 |

### Family violation rates

| Arm | Seed | Family | Violation rate |
| --- | --- | --- | --- |
| A_none | 0 | binder_arity | 0.000 |
| A_none | 0 | empty_vs_child | 0.000 |
| A_none | 0 | rare_component_substitution | 0.000 |
| A_none | 0 | same_type_different_role | 0.000 |
| A_none | 0 | slot_pointer | 0.000 |
| A_none | 0 | stack_vs_card | 0.000 |
| B_uniform | 0 | binder_arity | 0.500 |
| B_uniform | 0 | empty_vs_child | 0.750 |
| B_uniform | 0 | rare_component_substitution | 0.500 |
| B_uniform | 0 | same_type_different_role | 0.500 |
| B_uniform | 0 | slot_pointer | 0.750 |
| B_uniform | 0 | stack_vs_card | 1.000 |
| C_targeted_hardest | 0 | binder_arity | 0.500 |
| C_targeted_hardest | 0 | empty_vs_child | 0.750 |
| C_targeted_hardest | 0 | rare_component_substitution | 0.500 |
| C_targeted_hardest | 0 | same_type_different_role | 0.500 |
| C_targeted_hardest | 0 | slot_pointer | 0.750 |
| C_targeted_hardest | 0 | stack_vs_card | 1.000 |
| D_targeted_weighted | 0 | binder_arity | 0.500 |
| D_targeted_weighted | 0 | empty_vs_child | 0.750 |
| D_targeted_weighted | 0 | rare_component_substitution | 0.500 |
| D_targeted_weighted | 0 | same_type_different_role | 0.500 |
| D_targeted_weighted | 0 | slot_pointer | 0.750 |
| D_targeted_weighted | 0 | stack_vs_card | 1.000 |
| E_shuffled | 0 | binder_arity | 0.500 |
| E_shuffled | 0 | empty_vs_child | 0.500 |
| E_shuffled | 0 | rare_component_substitution | 1.000 |
| E_shuffled | 0 | same_type_different_role | 0.750 |
| E_shuffled | 0 | slot_pointer | 0.750 |
| E_shuffled | 0 | stack_vs_card | 0.500 |

## Go / no-go decision

**No-go for promotion.** Every arm is explicitly non-promotable. The harness proves the wiring and metrics plumbing over deterministic synthetic contrast rows, but it does not train or evaluate a real model. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained scorer and AgentV evaluation are available.

## Honest caveats

- Scores are deterministic hash-based dummy values, not a trained model.
- Contrast rows are synthetic and cover a hand-picked action vocabulary.
- Family weights are uniform in this fixture; real runs may load a   manifest with per-family weights.
- No Pareto or ship-gate claim is made; this is wiring evidence only.

## Reproducibility

```bash
python -m scripts.run_slm164_targeted_margin_fixture --mode plan-only
python -m scripts.run_slm164_targeted_margin_fixture --mode fixture
```
