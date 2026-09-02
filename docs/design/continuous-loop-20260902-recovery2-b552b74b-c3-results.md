# Continuous cycle `continuous-loop-20260902-recovery2-b552b74b-c3`

- loop_id: `recovery2`
- cycle_index: `3`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260902-recovery2-b552b74b-c3-control:smoke:invalid_counts:document_n=None:completed_document_n=None:incomplete_document_n=None:decode_timeout_count=None, measurement_incomplete:c20260902-recovery2-b552b74b-c3-semantic-contrast-compiler-margin:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': 4.568895543732056, 'smoke.eval_nll': 4.568895543732056}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, fit_decode_timeout_to_n_times_p50
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
