# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c519`

- loop_id: `continuous-openui-local`
- cycle_index: `519`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c519-control, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c519-current-rung-data-heal, non_regression_fail:binder_reference_f1:0.5333333333333333->0.0, primary_metric_null_or_worse:smoke.structural_similarity:control=0.13118333333333335 candidate=0.0534 improvement=-0.07778333333333334, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 14224.77, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.13118333333333335, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 14224.77, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.13118333333333335, 'smoke.binder_reference_f1': 0.5333333333333333}`
- candidate_metrics: `{'latency_ms_p50': 3115.92, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.0534, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 3115.92, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.0534, 'smoke.binder_reference_f1': 0.0}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
