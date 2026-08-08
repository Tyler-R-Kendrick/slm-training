# Continuous cycle `continuous-loop-20260808-continuous-openui-local-8c0b60dd-c2`

- loop_id: `continuous-openui-local`
- cycle_index: `2`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260808-continuous-openui-local-8c0b60dd-c2-control:smoke:incomplete_document_n=3:decode_timeout_count=3, harness_failure:c20260808-continuous-openui-local-8c0b60dd-c2-control:experiment_failed, fixture_insufficient_n:c20260808-continuous-openui-local-8c0b60dd-c2-component-plan, fixture_insufficient_n:c20260808-continuous-openui-local-8c0b60dd-c2-control, executable_unblock_rejected_low_mpr:mpr=0.0, primary_metric_unavailable, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- candidate_metrics: `{'latency_ms_p50': 32672.07, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.38280000000000003, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 32672.07, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.38280000000000003, 'smoke.binder_reference_f1': 0.0}`
- positive: **True**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-local-8c0b60dd-c2-component-plan, fixture_insufficient_n:c20260808-continuous-openui-local-8c0b60dd-c2-control, primary_metric_win:smoke.structural_similarity:0.32666666666666666->0.38280000000000003:improvement=0.05613333333333337
- control_metrics: `{'latency_ms_p50': 29097.75, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.32666666666666666, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 29097.75, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.32666666666666666, 'smoke.binder_reference_f1': 0.0}`
- candidate_metrics: `{'latency_ms_p50': 24161.42, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.38280000000000003, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 24161.42, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.38280000000000003, 'smoke.binder_reference_f1': 0.0}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
