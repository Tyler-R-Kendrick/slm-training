# Continuous cycle `continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2389`

- loop_id: `continuous-openui-local`
- cycle_index: `2389`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260830-continuous-openui-local-8c0b60dd-c2389-control:missing_scoreboard, measurement_incomplete:c20260830-continuous-openui-local-8c0b60dd-c2389-current-rung-data-heal:missing_scoreboard, harness_failure:c20260830-continuous-openui-local-8c0b60dd-c2389-control:experiment_failed, empty_metrics:686bb7c5e798d7ca13a09a270f83e537b13e942500cf9a9219af571111f3ad98, harness_failure:c20260830-continuous-openui-local-8c0b60dd-c2389-current-rung-data-heal:experiment_failed, empty_metrics:4d17447056f642bb86c14f3b11cc2f264e25cad8d5c1b987d9591c75e05b21b6, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
