# Continuous cycle `continuous-loop-20260808-continuous-openui-schedu-b09a9491-c3`

- loop_id: `continuous-openui-scheduled-0808b`
- cycle_index: `3`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260808-continuous-openui-schedu-b09a9491-c3-control:smoke:incomplete_document_n=3:decode_timeout_count=3, harness_failure:c20260808-continuous-openui-schedu-b09a9491-c3-control:experiment_failed, fixture_insufficient_n:c20260808-continuous-openui-schedu-b09a9491-c3-component-plan, fixture_insufficient_n:c20260808-continuous-openui-schedu-b09a9491-c3-control, executable_unblock_rejected_low_mpr:mpr=0.0, primary_metric_unavailable, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- candidate_metrics: `{'latency_ms_p50': 31673.63, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.38280000000000003, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 31673.63, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.38280000000000003, 'smoke.binder_reference_f1': 0.0}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
