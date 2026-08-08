# Continuous cycle `continuous-loop-20260808-continuous-openui-schedu-6e2cdb52-c3`

- loop_id: `continuous-openui-scheduled-w1tlbr`
- cycle_index: `3`
- role/intent: `screening` / `confirm`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-schedu-6e2cdb52-c3-control, fixture_insufficient_n:c20260808-continuous-openui-schedu-6e2cdb52-c3-confirm, efficiency_win_rejected_min_effect:mpr_per_ms:4.6879161e-05->4.8960639e-05:gain_fraction=0.044400935<0.05, non_regression_fail:binder_reference_f1:0.8222222222222223->0.6, primary_metric_null_or_worse:smoke.structural_similarity:control=0.19083333333333333 candidate=0.19083333333333333 improvement=0.0, fixture_insufficient_n_alone, confirmation_rejected:primary_quality_not_reheld
- control_metrics: `{'latency_ms_p50': 7110.48, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.19083333333333333, 'binder_reference_f1': 0.8222222222222223, 'smoke.latency_ms_p50': 7110.48, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.19083333333333333, 'smoke.binder_reference_f1': 0.8222222222222223}`
- candidate_metrics: `{'latency_ms_p50': 6808.19, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.19083333333333333, 'binder_reference_f1': 0.6, 'smoke.latency_ms_p50': 6808.19, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.19083333333333333, 'smoke.binder_reference_f1': 0.6}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
