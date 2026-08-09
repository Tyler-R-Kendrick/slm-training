# Continuous cycle `continuous-loop-20260808-continuous-openui-schedu-8d54410a-c3`

- loop_id: `continuous-openui-scheduled-0808c`
- cycle_index: `3`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260808-continuous-openui-schedu-8d54410a-c3-control:smoke:incomplete_document_n=3:decode_timeout_count=3, measurement_incomplete:c20260808-continuous-openui-schedu-8d54410a-c3-component-plan:smoke:incomplete_document_n=3:decode_timeout_count=3, harness_failure:c20260808-continuous-openui-schedu-8d54410a-c3-component-plan:experiment_failed, harness_failure:c20260808-continuous-openui-schedu-8d54410a-c3-control:experiment_failed, fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c3-control, fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c3-component-plan, primary_metric_unavailable, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- role/intent: `screening` / `confirm`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c3-confirm, fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c3-control, efficiency_win_rejected_min_effect:mpr_per_ms:4.4527027e-05->4.6738036e-05:gain_fraction=0.049655424<0.05, non_regression_fail:binder_reference_f1:0.8222222222222223->0.6, primary_metric_null_or_worse:smoke.structural_similarity:control=0.19083333333333333 candidate=0.19083333333333333 improvement=0.0, fixture_insufficient_n_alone, confirmation_rejected:primary_quality_not_reheld
- control_metrics: `{'latency_ms_p50': 7486.09, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.19083333333333333, 'binder_reference_f1': 0.8222222222222223, 'smoke.latency_ms_p50': 7486.09, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.19083333333333333, 'smoke.binder_reference_f1': 0.8222222222222223}`
- candidate_metrics: `{'latency_ms_p50': 7131.95, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.19083333333333333, 'binder_reference_f1': 0.6, 'smoke.latency_ms_p50': 7131.95, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.19083333333333333, 'smoke.binder_reference_f1': 0.6}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
