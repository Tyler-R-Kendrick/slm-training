# SLM-156 (SPV3-03): Shared recursive SemanticPlanV1 refinement (slm156_fixture)

Matrix set: `slm156_plan_refinement`

Version: `spv3-03-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no production TwoTower wiring was touched, and no ship-gate claim is made.

## Hypothesis

A small shared refinement cell applied recursively to SemanticPlanV1 improves plan-factor recovery over a parameter-matched one-pass predictor on coupled corruptions, and adaptive halting preserves quality at lower average depth.

## Falsifier

Recursion changes plans but not final correctness, deeper non-shared matches it at equal FLOPs, adaptive halting collapses to min/max depth, or diagnostics leak gold information.

## Arms

| Arm | Kind | Depth | Adaptive | Diagnostic | Description |
| --- | --- | --- | --- | --- | --- |
| A_one_pass | one_pass | 1 | False | False | Single forward through a parameter-matched one-pass predictor. |
| B_deeper_non_shared | deeper | 1 | False | False | Deeper non-shared predictor with matched parameter budget. |
| C_shared_fixed_2 | shared_fixed | 2 | False | False | Shared cell applied for exactly 2 recursions. |
| D_shared_fixed_4 | shared_fixed | 4 | False | False | Shared cell applied for exactly 4 recursions. |
| E_shared_adaptive | shared_adaptive | 4 | True | False | Shared cell with calibrated halt/value head. |
| F_shared_diagnostics | shared_adaptive | 4 | True | False | Adaptive shared cell plus inference-available diagnostics placeholder. |
| G_stochastic_value | shared_fixed | 4 | False | False | Stochastic trajectory sampling with value selection (fixture only). |
| H_gold_oracle | diagnostic | 0 | False | True | Gold plan oracle ceiling. |

## Results

| Arm | Seed | Records | Mean plan score | Mean depth | Mean forwards |
| --- | --- | --- | --- | --- | --- |
| A_one_pass | 0 | 8 | 0.697 | 0.0 | 1.0 |
| B_deeper_non_shared | 0 | 8 | 0.648 | 0.0 | 2.0 |
| C_shared_fixed_2 | 0 | 8 | 0.665 | 2.0 | 2.0 |
| D_shared_fixed_4 | 0 | 8 | 0.707 | 4.0 | 4.0 |
| E_shared_adaptive | 0 | 8 | 0.707 | 4.0 | 4.0 |
| F_shared_diagnostics | 0 | 8 | 0.707 | 4.0 | 4.0 |
| G_stochastic_value | 0 | 8 | 0.707 | 4.0 | 4.0 |
| H_gold_oracle | 0 | 8 | 0.771 | 0.0 | 0.0 |

## Verdict

This is a fixture wiring run. It validates that a shared refinement cell, fixed-depth and adaptive recursion, a deeper non-shared control, and an oracle ceiling can be registered under a common manifest with deterministic cost accounting. Real claims require a trained SemanticPlanV1 predictor, held-out causal downstream evaluation, and wall-clock measurement.
