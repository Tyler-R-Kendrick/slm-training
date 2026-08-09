# Continuous cycle `continuous-loop-20260808-continuous-openui-schedu-8d54410a-c4`

- loop_id: `continuous-openui-scheduled-0808c`
- cycle_index: `4`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260808-continuous-openui-schedu-8d54410a-c4-control:missing_scoreboard, measurement_incomplete:c20260808-continuous-openui-schedu-8d54410a-c4-bounds:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c4-control, fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c4-batch1, efficiency_win_rejected_min_effect:mpr_per_ms:1.5324747e-05->1.56815e-05:gain_fraction=0.0232795<0.05, primary_metric_null_or_worse:smoke.structural_similarity:control=0.4166666666666667 candidate=0.4166666666666667 improvement=0.0, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': 21751.31, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.4166666666666667, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 21751.31, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.4166666666666667, 'smoke.binder_reference_f1': 0.9523809523809524}`
- candidate_metrics: `{'latency_ms_p50': 21256.47, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.4166666666666667, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 21256.47, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.4166666666666667, 'smoke.binder_reference_f1': 0.9523809523809524}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
