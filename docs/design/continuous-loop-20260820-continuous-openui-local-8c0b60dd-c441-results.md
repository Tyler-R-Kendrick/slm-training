# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c441`

- loop_id: `continuous-openui-local`
- cycle_index: `441`
- role/intent: `screening` / `confirm`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c441-control, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c441-confirm, non_regression_fail:binder_reference_f1:0.6333333333333333->0.0, primary_metric_null_or_worse:smoke.structural_similarity:control=0.13526666666666667 candidate=0.36000000000000004 improvement=0.22473333333333337, fixture_insufficient_n_alone, confirmation_rejected:primary_quality_not_reheld
- control_metrics: `{'latency_ms_p50': 2807.21, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.13526666666666667, 'binder_reference_f1': 0.6333333333333333, 'smoke.latency_ms_p50': 2807.21, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.13526666666666667, 'smoke.binder_reference_f1': 0.6333333333333333}`
- candidate_metrics: `{'latency_ms_p50': 10432.47, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.36000000000000004, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 10432.47, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.36000000000000004, 'smoke.binder_reference_f1': 0.0}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
