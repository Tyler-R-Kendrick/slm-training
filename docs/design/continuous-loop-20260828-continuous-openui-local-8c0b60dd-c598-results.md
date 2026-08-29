# Continuous cycle `continuous-loop-20260828-continuous-openui-local-8c0b60dd-c598`

- loop_id: `continuous-openui-local`
- cycle_index: `598`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260828-continuous-openui-local-8c0b60dd-c598-control:missing_scoreboard, fixture_insufficient_n:c20260828-continuous-openui-local-8c0b60dd-c598-current-rung-data-heal-latprobe, primary_metric_unavailable, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': 16079.42, 'parse_rate': 1.0, 'meaningful_program_rate': 0.375, 'structural_similarity': 0.24786666666666668, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': 7.327639480481518, 'smoke.latency_ms_p50': 16079.42, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.375, 'smoke.structural_similarity': 0.24786666666666668, 'smoke.binder_reference_f1': 0.5313492063492063, 'smoke.eval_nll': 7.327639480481518}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
