# Continuous cycle `continuous-loop-20260808-continuous-openui-schedu-8d54410a-c5`

- loop_id: `continuous-openui-scheduled-0808c`
- cycle_index: `5`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c5-bounds, fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c5-control, primary_metric_null_or_worse:smoke.structural_similarity:control=0.4166666666666667 candidate=0.4166666666666667 improvement=0.0, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': 30314.2, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.4166666666666667, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 30314.2, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.4166666666666667, 'smoke.binder_reference_f1': 0.9523809523809524}`
- candidate_metrics: `{'latency_ms_p50': 31308.1, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.4166666666666667, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 31308.1, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.4166666666666667, 'smoke.binder_reference_f1': 0.9523809523809524}`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260808-continuous-openui-schedu-8d54410a-c5-component-plan:smoke:incomplete_document_n=3:decode_timeout_count=3, fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c5-control, fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c5-component-plan, primary_metric_unavailable, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': 14108.67, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.057499999999999996, 'binder_reference_f1': 0.8222222222222223, 'smoke.latency_ms_p50': 14108.67, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.057499999999999996, 'smoke.binder_reference_f1': 0.8222222222222223}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
