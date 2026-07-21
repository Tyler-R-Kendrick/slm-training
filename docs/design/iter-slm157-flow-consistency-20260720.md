# SLM-157 (SPV3-04): Flow / consistency / trajectory-imitation fixture (slm157-flow-consistency-20260720)

Matrix set: `slm157_flow_consistency`

Version: `spv3-04-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no production TwoTower wiring was touched, and no ship-gate claim is made.

## Hypothesis

Discrete-flow, consistency, and trajectory-imitation policies can be simulated over the hard-valid tree-edit state space using only the existing compiler and verified patch surfaces, before any learned scorer is trained.

## Falsifier

The existing legal-edit enumeration cannot produce non-trivial paths between distinct valid programs, the consistency boundary signal is indistinguishable from the greedy distance signal, or every synthetic arm collapses to the same trivial trajectory.

## Arms

| Arm | Family | Path family | Promotable | Diagnostic | Description |
| --- | --- | --- | --- | --- | --- |
| A_teacher_long_x22 | teacher_long_x22 | P_x22 | False | False | Long-horizon X22 teacher: greedy distance-reducing walk over the full legal-edit enumeration. |
| B_direct_trajectory_imitation | direct_trajectory_imitation | P_short | False | False | Direct trajectory imitation: follow the short verified patch path from source to target. |
| C_consistency_student_x22 | consistency_student_x22 | P_x22 | False | False | Consistency student trained on the X22 path family: boundary-state match using the full edit distance. |
| D_consistency_student_coarse | consistency_student_coarse | P_coarse | False | False | Consistency student trained on the coarse path family: boundary-state match using a coarse component-expression distance. |
| E_discrete_flow_rate | discrete_flow_rate | P_capsule | False | False | Discrete flow-rate policy: softmax over negative remaining distance on a capsule-shaped reference path. |
| F_random_path_control | random_path_control | P_random | False | False | Random-path control: uniform random legal edits; sanity-check baseline for reach rates. |
| G_ar_x22_hybrid_placeholder | ar_x22_hybrid_placeholder | P_x22 | False | False | AR/X22 hybrid placeholder: policy wiring only, not a trained autoregressive scorer. |
| H_oracle_boundary | oracle_boundary | P_short | False | True | Oracle boundary diagnostic: perfect boundary prediction and short-path navigation; upper-bound sanity check. |

## Results

| Arm | Seed | Steps | Records | Target reach | Accepted reach | Boundary acc | Path len | Remaining dist | Detour | Consistency | Entropy | Rollbacks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_teacher_long_x22 | 0 | 4 | 2 | 0.000 | 0.000 | 1.000 | 1.50 | 1.50 | 0.38 | 0.500 | 0.000 | 0.000 |
| B_direct_trajectory_imitation | 0 | 4 | 2 | 0.000 | 0.000 | 0.000 | 0.00 | 3.00 | 0.00 | 0.000 | 0.000 | 0.000 |
| C_consistency_student_x22 | 0 | 4 | 2 | 0.000 | 0.000 | 1.000 | 1.50 | 1.50 | 0.38 | 0.500 | 0.000 | 0.000 |
| D_consistency_student_coarse | 0 | 4 | 2 | 0.000 | 0.000 | 1.000 | 1.50 | 2.00 | 0.38 | 0.500 | 0.000 | 0.000 |
| E_discrete_flow_rate | 0 | 4 | 2 | 0.000 | 0.000 | 1.000 | 4.00 | 4.00 | 1.50 | 0.750 | 3.617 | 0.000 |
| F_random_path_control | 0 | 4 | 2 | 0.000 | 0.000 | 1.000 | 4.00 | 5.00 | 1.50 | 0.375 | 3.849 | 0.000 |
| G_ar_x22_hybrid_placeholder | 0 | 4 | 2 | 0.000 | 0.000 | 1.000 | 4.00 | 3.50 | 1.50 | 0.500 | 0.000 | 0.000 |
| H_oracle_boundary | 0 | 4 | 2 | 0.000 | 0.000 | 0.375 | 1.50 | 1.50 | 0.38 | 0.500 | 0.000 | 0.000 |

## Go / no-go decision

**No-go for promotion.** Every arm is explicitly non-promotable. The harness proves the wiring and metrics plumbing over synthetic, hard-valid trajectories, but it does not train or evaluate a real model. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained scorer and AgentV evaluation are available.

## Honest caveats

- The source/target pairs come from a small fixture plan corpus, not a   production train/eval split.
- Distance and reach metrics use the existing statement-level patch   distance, not a rendering or user-judgment proxy.
- Boundary prediction is synthetic: STOP is treated as a boundary   prediction, and accuracy is measured against the known target.
- Rollbacks are recorded when the selected edit is invalid, but the   legal-edit enumeration filters invalid candidates before selection.
- No Pareto or ship-gate claim is made; this is wiring evidence only.

## Reproducibility

```bash
python -m scripts.run_slm157_flow_consistency_fixture --mode plan-only
python -m scripts.run_slm157_flow_consistency_fixture --mode fixture
```
