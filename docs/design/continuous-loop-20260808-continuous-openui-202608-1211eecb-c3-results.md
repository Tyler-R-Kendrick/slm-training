# Continuous cycle `continuous-loop-20260808-continuous-openui-202608-1211eecb-c3`

- loop_id: `continuous-openui-20260808`
- cycle_index: `3`
- role/intent: `screening` / `confirm`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c3-control, fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c3-confirm, non_regression_fail:binder_reference_f1:0.8222222222222223->0.6, primary_metric_null_or_worse:smoke.structural_similarity:control=0.19083333333333333 candidate=0.19083333333333333 improvement=0.0, fixture_insufficient_n_alone, confirmation_rejected:primary_quality_not_reheld
- control_metrics: `{'latency_ms_p50': 8194.52, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.19083333333333333, 'binder_reference_f1': 0.8222222222222223, 'smoke.latency_ms_p50': 8194.52, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.19083333333333333, 'smoke.binder_reference_f1': 0.8222222222222223}`
- candidate_metrics: `{'latency_ms_p50': 8668.02, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.19083333333333333, 'binder_reference_f1': 0.6, 'smoke.latency_ms_p50': 8668.02, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.19083333333333333, 'smoke.binder_reference_f1': 0.6}`
- control_metrics: `{'latency_ms_p50': 7779.06, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.19083333333333333, 'binder_reference_f1': 0.8222222222222223, 'smoke.latency_ms_p50': 7779.06, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.19083333333333333, 'smoke.binder_reference_f1': 0.8222222222222223}`
- candidate_metrics: `{'latency_ms_p50': 8717.13, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.19083333333333333, 'binder_reference_f1': 0.6, 'smoke.latency_ms_p50': 8717.13, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.19083333333333333, 'smoke.binder_reference_f1': 0.6}`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260808-continuous-openui-202608-1211eecb-c3-control:smoke:incomplete_document_n=3:decode_timeout_count=3, harness_failure:c20260808-continuous-openui-202608-1211eecb-c3-control:experiment_failed, fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c3-control, fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c3-component-plan, executable_unblock_rejected_low_mpr:mpr=0.0, primary_metric_unavailable, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- candidate_metrics: `{'latency_ms_p50': 31909.74, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.38280000000000003, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 31909.74, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.38280000000000003, 'smoke.binder_reference_f1': 0.0}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
