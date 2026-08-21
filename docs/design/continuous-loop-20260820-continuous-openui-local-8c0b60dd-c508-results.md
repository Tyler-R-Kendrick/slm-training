# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c508`

- loop_id: `continuous-openui-local`
- cycle_index: `508`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c508-current-rung-data-heal, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c508-control, non_regression_fail:binder_reference_f1:0.8555555555555555->0.7619047619047619, primary_metric_null_or_worse:smoke.structural_similarity:control=0.20898333333333333 candidate=0.18868333333333331 improvement=-0.020300000000000012, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 26101.36, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.20898333333333333, 'binder_reference_f1': 0.8555555555555555, 'smoke.latency_ms_p50': 26101.36, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.20898333333333333, 'smoke.binder_reference_f1': 0.8555555555555555}`
- candidate_metrics: `{'latency_ms_p50': 13049.48, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.18868333333333331, 'binder_reference_f1': 0.7619047619047619, 'smoke.latency_ms_p50': 13049.48, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.18868333333333331, 'smoke.binder_reference_f1': 0.7619047619047619}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': -0.09365079365079365, 'latency_ms_p50': -13051.880000000001, 'meaningful_program_rate': 0.0, 'parse_rate': 0.0, 'smoke.binder_reference_f1': -0.09365079365079365, 'smoke.latency_ms_p50': -13051.880000000001, 'smoke.meaningful_program_rate': 0.0, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': -0.020300000000000012, 'structural_similarity': -0.020300000000000012}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
