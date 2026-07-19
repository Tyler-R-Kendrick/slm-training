# SDE4-01 (SLM-179) scaffold-distillation activation manifest

**manifest_id:** `scaffold-distillation-activation-20260719`  
**schema_version:** `scaffold_distillation_activation/v1`  
**hypothesis_id:** `H13`  
**activation_status:** `ready`  
**activation_verdict:** `ready_to_spend`  
**campaign_verdict:** `unrun`  
**manifest_hash:** `281dc33bee9b92a9`  
**primary_metric:** `binding_aware_meaningful_program_rate`  
**max_attempts_for_teacher:** 1  
**seeds:** [0, 1, 2]

## Activation gates

| gate_id | depends_on_issue_id | required_status | available | evidence |
| --- | --- | --- | ---: | --- |
| slm161_machine_readable_decomposition | SLM-161 | Done | True | Machine-readable decomposition of the scaffold into levered operations. |
| slm162_metric_gaming_suite | SLM-162 | Done | True | Metric-gaming stress suite demonstrates the scaffold is not gamed. |
| slm168_public_structured_contract_pointer | SLM-168 | Done | True | Public structured contract pointer exposes the inventory gap. |
| scaffold_value_demonstrated | SLM-161 | Done | True | The scaffold improves over no-scaffold baselines on a held slice. |
| latency_or_complexity_worth_amortizing |  |  | True | Teacher scaffolding cost is worth amortizing over student training. |
| budget_approved |  |  | True | Campaign budget approved for teacher traces and student runs. |

## Budget cap

| teacher_trace_compute_dollars | student_training_dollars | student_training_gpu_hours | eval_dollars | total_dollars |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0 | 0.0 | 0.0 | 1000.0 |

## Teacher trace contract

| teacher_checkpoint_id | teacher_run_id | trace_store_uri | trace_schema_version | min_traces | max_traces | scaffold_config_hash |
| --- | --- | --- | --- | ---: | ---: | --- |
| teacher/checkpoint | teacher/run | memory://traces | v1 | 0 | 0 | unknown |

## Arms

| arm_id | arm_kind | eligible | objectives | omission_reason |
| --- | --- | ---: | --- | --- |
| scaffolded_teacher_selected | scaffolded_teacher_selected | True |  |  |
| teacher_first_attempt_only | teacher_first_attempt_only | True |  |  |
| lever_off_gold_sft | lever_off_gold_sft | True |  |  |
| selected_trajectory_distillation | selected_trajectory_distillation | True | sft |  |
| sft_plus_legal_set_kl | sft_plus_legal_set_kl | True | sft, kl |  |
| sft_kl_plus_preference | sft_kl_plus_preference | True | sft, kl, preference |  |
| permuted_teacher_specificity_control | permuted_teacher_specificity_control | True | sft |  |
| impossible_information_inventory_control | impossible_information_inventory_control | False |  | non-promotable control only; quantifies the information gap when the student cannot access the teacher inventory |

## Note

SDE4-01 (SLM-179) scaffold-distillation activation manifest (wiring slice).

Full detail: `docs/design/iter-scaffold-distillation-activation-20260719.json`.
