# Continuous cycle `continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2388`

- loop_id: `continuous-openui-local`
- cycle_index: `2388`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260830-continuous-openui-local-8c0b60dd-c2388-control:missing_scoreboard, measurement_incomplete:c20260830-continuous-openui-local-8c0b60dd-c2388-current-rung-data-heal:missing_scoreboard, harness_failure:c20260830-continuous-openui-local-8c0b60dd-c2388-control:experiment_failed, empty_metrics:69851c93904d2e761adec2560a31de5cece0ce3293e370c02bac9c50e1ceb1c7, harness_failure:c20260830-continuous-openui-local-8c0b60dd-c2388-current-rung-data-heal:experiment_failed, empty_metrics:ac58710929274cc0399047300be5f66bb7a6cde026ade2d7e6495b70aa5ac590, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
