# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c511`

- loop_id: `continuous-openui-local`
- cycle_index: `511`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260820-continuous-openui-local-8c0b60dd-c511-control:missing_scoreboard, measurement_incomplete:c20260820-continuous-openui-local-8c0b60dd-c511-current-rung-data-heal:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
