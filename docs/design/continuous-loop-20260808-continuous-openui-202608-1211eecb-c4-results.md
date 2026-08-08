# Continuous cycle `continuous-loop-20260808-continuous-openui-202608-1211eecb-c4`

- loop_id: `continuous-openui-20260808`
- cycle_index: `4`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c4-batch1, fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c4-control, primary_metric_null_or_worse:smoke.structural_similarity:control=0.4166666666666667 candidate=0.4166666666666667 improvement=0.0, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': 24309.46, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.4166666666666667, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 24309.46, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.4166666666666667, 'smoke.binder_reference_f1': 0.9523809523809524}`
- candidate_metrics: `{'latency_ms_p50': 24761.6, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.4166666666666667, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 24761.6, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.4166666666666667, 'smoke.binder_reference_f1': 0.9523809523809524}`
- control_metrics: `{'latency_ms_p50': 23312.38, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.4166666666666667, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 23312.38, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.4166666666666667, 'smoke.binder_reference_f1': 0.9523809523809524}`
- candidate_metrics: `{'latency_ms_p50': 25507.37, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.4166666666666667, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 25507.37, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.4166666666666667, 'smoke.binder_reference_f1': 0.9523809523809524}`
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260808-continuous-openui-202608-1211eecb-c4-control:smoke:incomplete_document_n=3:decode_timeout_count=3, harness_failure:c20260808-continuous-openui-202608-1211eecb-c4-control:experiment_failed, fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c4-bounds, fixture_insufficient_n:c20260808-continuous-openui-202608-1211eecb-c4-control, primary_metric_unavailable, executable_unblock:candidate_completed_after_control_error
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- candidate_metrics: `{'latency_ms_p50': 30832.89, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.4166666666666667, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 30832.89, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.4166666666666667, 'smoke.binder_reference_f1': 0.9523809523809524}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
