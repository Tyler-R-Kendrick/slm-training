# Continuous cycle `continuous-loop-20260902-recovery2-b552b74b-c2`

- loop_id: `recovery2`
- cycle_index: `2`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260902-recovery2-b552b74b-c2-control:missing_scoreboard, measurement_incomplete:c20260902-recovery2-b552b74b-c2-semantic-contrast-compiler-margin:missing_scoreboard, harness_failure:c20260902-recovery2-b552b74b-c2-semantic-contrast-compiler-margin:experiment_failed, empty_metrics:3894a2ab2ca4dea245d8ea121dfc46f5e476875427301571cae10505faa8dc6c, harness_failure:c20260902-recovery2-b552b74b-c2-control:experiment_failed, empty_metrics:4714f69c06d1d35dff7b0bf5ff5c7a3f00208f05043790ab32a3eb182846b258, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
