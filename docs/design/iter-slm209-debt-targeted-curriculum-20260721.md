# SLM-209 (SDE5-02): debt-targeted semantic exposure curriculum fixture (slm209-debt-targeted-curriculum-20260721)

Matrix set: `slm209_debt_targeted_curriculum`

Version: `sde5-02-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

A fixed-budget curriculum that selects exact states by grammar-mask debt, inverse decision-kind frequency, and legal-support entropy increases the exposure of high-debt exact states while respecting per-group caps and train/held-out isolation.

## Falsifier

Debt-targeted selection fails to increase high-debt state exposure, violates the total decision budget, exceeds per-group caps, or leaks a group across train/held-out splits.

## Cells

| policy_name | seed | decision_budget | per_group_cap | selected_states | unique_groups | max_group_count | mean_effective_debt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 120 | 6 | 60 | 40 | 2 | 1.8956 |
| slm170_frequency | 0 | 120 | 6 | 60 | 40 | 2 | 1.8956 |
| high_debt | 0 | 120 | 6 | 60 | 40 | 2 | 1.8956 |
| debt_plus_rarity | 0 | 120 | 6 | 60 | 40 | 2 | 1.8956 |
| debt_plus_entropy | 0 | 120 | 6 | 60 | 40 | 2 | 1.8956 |
| preregistered_composite | 0 | 120 | 6 | 60 | 40 | 2 | 1.8956 |

## Exposure audit (preregistered composite example)

- By decision kind: {'argument_value': 7, 'array_insert': 10, 'component_choice': 7, 'constraint_shadow': 18, 'root_closure': 10, 'slot_binding': 8}
- By split: {'held_out': 16, 'train': 44}
- Unique groups: 40
- Max group count: 2
- Mean selection score: 1.2244
- Mean effective debt: 1.8956

## Disposition

**no_debt_lift**

Debt-targeted policies do not select higher-debt states than uniform sampling in this synthetic fixture.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The selection policies, score components, and caps are exercised over deterministic synthetic states, but no real model was trained or evaluated. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until trained-model constraint-debt telemetry and AgentV evaluation are available.

## Honest caveats

- Synthetic fixture: no model, checkpoint, or verifier labels were used.
- Debt masses are randomly generated and only weakly correlate with decision kind; real constraint-debt telemetry will differ.
- Per-group caps are enforced at the group_id level; real curricula may need trajectory-level or program-family-level caps.
- No ship-gate claim is made; this is wiring evidence only.

## Reproducibility

```bash
python -m scripts.run_slm209_debt_targeted_curriculum_fixture --mode plan-only
python -m scripts.run_slm209_debt_targeted_curriculum_fixture --mode fixture
```
