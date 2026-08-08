# Continuous cycle `continuous-loop-20260808-continuous-openui-202608-13c4d22a-c3`

- loop_id: `continuous-openui-20260808-scheduled`
- cycle_index: `3`
- role/intent: `screening` / `confirm`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-202608-13c4d22a-c3-confirm, fixture_insufficient_n:c20260808-continuous-openui-202608-13c4d22a-c3-control, non_regression_fail:binder_reference_f1:0.8222222222222223->0.6, primary_metric_null_or_worse:smoke.structural_similarity:control=0.19083333333333333 candidate=0.19083333333333333 improvement=0.0, fixture_insufficient_n_alone, confirmation_rejected:primary_quality_not_reheld
- control_metrics: `{'latency_ms_p50': 7729.49, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.19083333333333333, 'binder_reference_f1': 0.8222222222222223, 'smoke.latency_ms_p50': 7729.49, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.19083333333333333, 'smoke.binder_reference_f1': 0.8222222222222223}`
- candidate_metrics: `{'latency_ms_p50': 7800.69, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.19083333333333333, 'binder_reference_f1': 0.6, 'smoke.latency_ms_p50': 7800.69, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.19083333333333333, 'smoke.binder_reference_f1': 0.6}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
