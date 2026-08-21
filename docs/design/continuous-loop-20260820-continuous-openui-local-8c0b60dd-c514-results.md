# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c514`

- loop_id: `continuous-openui-local`
- cycle_index: `514`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c514-current-rung-data-heal, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c514-control, quality_win_rejected_latency_budget:mpr=0.0->0.16666666666666666 lat=3300.5->10753.73, primary_metric_win:smoke.structural_similarity:0.07285->0.30446666666666666:improvement=0.23161666666666667, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 3300.5, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.07285, 'binder_reference_f1': 0.7666666666666666, 'smoke.latency_ms_p50': 3300.5, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.07285, 'smoke.binder_reference_f1': 0.7666666666666666}`
- candidate_metrics: `{'latency_ms_p50': 10753.73, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.30446666666666666, 'binder_reference_f1': 0.7666666666666666, 'smoke.latency_ms_p50': 10753.73, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.30446666666666666, 'smoke.binder_reference_f1': 0.7666666666666666}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': 0.0, 'latency_ms_p50': 7453.23, 'meaningful_program_rate': 0.16666666666666666, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 7453.23, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': 0.23161666666666667, 'structural_similarity': 0.23161666666666667}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
