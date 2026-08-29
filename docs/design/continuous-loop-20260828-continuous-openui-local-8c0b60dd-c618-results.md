# Continuous cycle `continuous-loop-20260828-continuous-openui-local-8c0b60dd-c618`

- loop_id: `continuous-openui-local`
- cycle_index: `618`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260828-continuous-openui-local-8c0b60dd-c618-control-latprobe, fixture_insufficient_n:c20260828-continuous-openui-local-8c0b60dd-c618-current-rung-data-heal-latprobe, primary_metric_unavailable, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 28650.67, 'parse_rate': 1.0, 'meaningful_program_rate': 0.4166666666666667, 'structural_similarity': 0.268975, 'binder_reference_f1': 0.7256944444444445, 'eval_nll': None, 'smoke.latency_ms_p50': 28650.67, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.4166666666666667, 'smoke.structural_similarity': 0.268975, 'smoke.binder_reference_f1': 0.7256944444444445}`
- candidate_metrics: `{'latency_ms_p50': 30778.92, 'parse_rate': 1.0, 'meaningful_program_rate': 0.4166666666666667, 'structural_similarity': 0.2450125, 'binder_reference_f1': 0.9465277777777779, 'eval_nll': None, 'smoke.latency_ms_p50': 30778.92, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.4166666666666667, 'smoke.structural_similarity': 0.2450125, 'smoke.binder_reference_f1': 0.9465277777777779}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': 0.22083333333333333, 'latency_ms_p50': 2128.25, 'meaningful_program_rate': 0.0, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.22083333333333333, 'smoke.latency_ms_p50': 2128.25, 'smoke.meaningful_program_rate': 0.0, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': -0.023962500000000025, 'structural_similarity': -0.023962500000000025}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
