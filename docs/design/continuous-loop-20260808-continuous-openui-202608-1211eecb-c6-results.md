# Continuous cycle `continuous-loop-20260808-continuous-openui-202608-1211eecb-c6`

- loop_id: `continuous-openui-20260808`
- cycle_index: `6`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260808-continuous-openui-202608-1211eecb-c6-component-plan:smoke:incomplete_document_n=3:decode_timeout_count=3, harness_failure:c20260808-continuous-openui-202608-1211eecb-c6-component-plan:experiment_failed, fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c6-component-plan, fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c6-control, primary_metric_unavailable, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': 13816.67, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.057499999999999996, 'binder_reference_f1': 0.8222222222222223, 'smoke.latency_ms_p50': 13816.67, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.057499999999999996, 'smoke.binder_reference_f1': 0.8222222222222223}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c6-control, fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c6-bounds, primary_metric_null_or_worse:smoke.structural_similarity:control=0.0964 candidate=0.0964 improvement=0.0, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': 3642.71, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.0964, 'binder_reference_f1': 0.8222222222222223, 'smoke.latency_ms_p50': 3642.71, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.0964, 'smoke.binder_reference_f1': 0.8222222222222223}`
- candidate_metrics: `{'latency_ms_p50': 3506.26, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.0964, 'binder_reference_f1': 0.8222222222222223, 'smoke.latency_ms_p50': 3506.26, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.0964, 'smoke.binder_reference_f1': 0.8222222222222223}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
