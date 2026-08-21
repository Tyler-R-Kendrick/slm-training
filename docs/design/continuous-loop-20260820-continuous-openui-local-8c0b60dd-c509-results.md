# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c509`

- loop_id: `continuous-openui-local`
- cycle_index: `509`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c509-control, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c509-current-rung-data-heal, quality_win_rejected_latency_budget:mpr=0.16666666666666666->0.3333333333333333 lat=3134.08->4970.37, primary_metric_win:smoke.structural_similarity:0.11561666666666666->0.13116666666666668:improvement=0.015550000000000022, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 3134.08, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.11561666666666666, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 3134.08, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.11561666666666666, 'smoke.binder_reference_f1': 0.5333333333333333}`
- candidate_metrics: `{'latency_ms_p50': 4970.37, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.13116666666666668, 'binder_reference_f1': 0.7666666666666666, 'smoke.latency_ms_p50': 4970.37, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.13116666666666668, 'smoke.binder_reference_f1': 0.7666666666666666}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': 0.23333333333333328, 'latency_ms_p50': 1836.29, 'meaningful_program_rate': 0.16666666666666666, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.23333333333333328, 'smoke.latency_ms_p50': 1836.29, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': 0.015550000000000022, 'structural_similarity': 0.015550000000000022}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
