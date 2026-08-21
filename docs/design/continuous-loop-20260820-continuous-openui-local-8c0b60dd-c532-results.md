# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c532`

- loop_id: `continuous-openui-local`
- cycle_index: `532`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c532-control, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c532-current-rung-data-heal, quality_win_rejected_latency_budget:mpr=0.0->0.3333333333333333 lat=3943.76->13828.36, primary_metric_win:smoke.structural_similarity:0.0534->0.2986333333333333:improvement=0.2452333333333333, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 3943.76, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.0534, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 3943.76, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.0534, 'smoke.binder_reference_f1': 0.0}`
- candidate_metrics: `{'latency_ms_p50': 13828.36, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.2986333333333333, 'binder_reference_f1': 1.0, 'smoke.latency_ms_p50': 13828.36, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.2986333333333333, 'smoke.binder_reference_f1': 1.0}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': 1.0, 'latency_ms_p50': 9884.6, 'meaningful_program_rate': 0.3333333333333333, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 1.0, 'smoke.latency_ms_p50': 9884.6, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': 0.2452333333333333, 'structural_similarity': 0.2452333333333333}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
