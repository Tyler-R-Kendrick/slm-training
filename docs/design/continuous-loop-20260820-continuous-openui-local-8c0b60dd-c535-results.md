# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c535`

- loop_id: `continuous-openui-local`
- cycle_index: `535`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c535-control, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c535-current-rung-data-heal, non_regression_fail:binder_reference_f1:1.0->0.7666666666666666, primary_metric_null_or_worse:smoke.structural_similarity:control=0.30451666666666666 candidate=0.07285 improvement=-0.23166666666666666, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 13774.56, 'parse_rate': 1.0, 'meaningful_program_rate': 0.5, 'structural_similarity': 0.30451666666666666, 'binder_reference_f1': 1.0, 'smoke.latency_ms_p50': 13774.56, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.5, 'smoke.structural_similarity': 0.30451666666666666, 'smoke.binder_reference_f1': 1.0}`
- candidate_metrics: `{'latency_ms_p50': 3137.14, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.07285, 'binder_reference_f1': 0.7666666666666666, 'smoke.latency_ms_p50': 3137.14, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.07285, 'smoke.binder_reference_f1': 0.7666666666666666}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
