# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c526`

- loop_id: `continuous-openui-local`
- cycle_index: `526`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c526-control, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c526-current-rung-data-heal, primary_metric_null_or_worse:smoke.structural_similarity:control=0.19063333333333332 candidate=0.13118333333333335 improvement=-0.059449999999999975, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 7544.7, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.19063333333333332, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 7544.7, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.19063333333333332, 'smoke.binder_reference_f1': 0.5333333333333333}`
- candidate_metrics: `{'latency_ms_p50': 3384.26, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.13118333333333335, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 3384.26, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.13118333333333335, 'smoke.binder_reference_f1': 0.5333333333333333}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': 0.0, 'latency_ms_p50': -4160.44, 'meaningful_program_rate': 0.0, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.0, 'smoke.latency_ms_p50': -4160.44, 'smoke.meaningful_program_rate': 0.0, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': -0.059449999999999975, 'structural_similarity': -0.059449999999999975}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
