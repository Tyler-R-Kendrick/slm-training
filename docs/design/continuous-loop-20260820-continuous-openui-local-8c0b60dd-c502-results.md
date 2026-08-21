# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c502`

- loop_id: `continuous-openui-local`
- cycle_index: `502`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c502-current-rung-data-heal, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c502-control, quality_win_rejected_latency_budget:mpr=0.0->0.16666666666666666 lat=4973.07->7424.95, primary_metric_win:smoke.structural_similarity:0.0534->0.13118333333333335:improvement=0.07778333333333334, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 4973.07, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.0534, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 4973.07, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.0534, 'smoke.binder_reference_f1': 0.0}`
- candidate_metrics: `{'latency_ms_p50': 7424.95, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.13118333333333335, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 7424.95, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.13118333333333335, 'smoke.binder_reference_f1': 0.5333333333333333}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': 0.5333333333333333, 'latency_ms_p50': 2451.88, 'meaningful_program_rate': 0.16666666666666666, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 2451.88, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': 0.07778333333333334, 'structural_similarity': 0.07778333333333334}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
