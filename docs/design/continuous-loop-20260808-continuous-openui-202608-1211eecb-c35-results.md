# Continuous cycle `continuous-loop-20260808-continuous-openui-202608-1211eecb-c35`

- loop_id: `continuous-openui-20260808`
- cycle_index: `35`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c35-semantic-contrast-compiler-margin, fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c35-control, non_regression_fail:binder_reference_f1:1.0->0.9523809523809524, primary_metric_null_or_worse:smoke.structural_similarity:control=0.0387 candidate=0.46973333333333334 improvement=0.4310333333333333, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': 25483.73, 'parse_rate': 1.0, 'meaningful_program_rate': 1.0, 'structural_similarity': 0.0387, 'binder_reference_f1': 1.0, 'smoke.latency_ms_p50': 25483.73, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 1.0, 'smoke.structural_similarity': 0.0387, 'smoke.binder_reference_f1': 1.0}`
- candidate_metrics: `{'latency_ms_p50': 24930.96, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.46973333333333334, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 24930.96, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.46973333333333334, 'smoke.binder_reference_f1': 0.9523809523809524}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
