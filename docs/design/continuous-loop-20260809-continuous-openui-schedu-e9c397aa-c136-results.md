# Continuous cycle `continuous-loop-20260809-continuous-openui-schedu-e9c397aa-c136`

- loop_id: `continuous-openui-scheduled-2805`
- cycle_index: `136`
- role/intent: `screening` / `confirm`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260809-continuous-openui-schedu-e9c397aa-c136-confirm, fixture_insufficient_n:c20260809-continuous-openui-schedu-e9c397aa-c136-control, non_regression_fail:binder_reference_f1:0.9523809523809524->0.7999999999999999, primary_metric_null_or_worse:smoke.structural_similarity:control=0.19083333333333333 candidate=0.06513333333333333 improvement=-0.12569999999999998, fixture_insufficient_n_alone, eg_params_block:capacity growth blocked for climb/promotion: capacity_increased_without_parameter_efficiency_evidence, confirmation_rejected:primary_quality_not_reheld
- control_metrics: `{'latency_ms_p50': 10279.24, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.19083333333333333, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 10279.24, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.19083333333333333, 'smoke.binder_reference_f1': 0.9523809523809524}`
- candidate_metrics: `{'latency_ms_p50': 12308.36, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.06513333333333333, 'binder_reference_f1': 0.7999999999999999, 'smoke.latency_ms_p50': 12308.36, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.06513333333333333, 'smoke.binder_reference_f1': 0.7999999999999999}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
