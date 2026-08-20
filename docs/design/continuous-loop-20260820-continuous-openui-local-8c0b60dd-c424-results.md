# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c424`

- loop_id: `continuous-openui-local`
- cycle_index: `424`
- role/intent: `screening` / `confirm`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c424-control, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c424-confirm, non_regression_fail:binder_reference_f1:0.6333333333333333->0.0, primary_metric_null_or_worse:smoke.structural_similarity:control=0.17416666666666666 candidate=0.36000000000000004 improvement=0.18583333333333338, fixture_insufficient_n_alone, confirmation_rejected:primary_quality_not_reheld
- control_metrics: `{'latency_ms_p50': 2665.99, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.17416666666666666, 'binder_reference_f1': 0.6333333333333333, 'smoke.latency_ms_p50': 2665.99, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.17416666666666666, 'smoke.binder_reference_f1': 0.6333333333333333}`
- candidate_metrics: `{'latency_ms_p50': 5832.79, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.36000000000000004, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 5832.79, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.36000000000000004, 'smoke.binder_reference_f1': 0.0}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
