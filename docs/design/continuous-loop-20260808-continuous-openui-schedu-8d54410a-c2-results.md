# Continuous cycle `continuous-loop-20260808-continuous-openui-schedu-8d54410a-c2`

- loop_id: `continuous-openui-scheduled-0808c`
- cycle_index: `2`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260808-continuous-openui-schedu-8d54410a-c2-control:missing_scoreboard, measurement_incomplete:c20260808-continuous-openui-schedu-8d54410a-c2-component-plan:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- positive: **True**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c2-component-plan, fixture_insufficient_n:c20260808-continuous-openui-schedu-8d54410a-c2-control, primary_metric_win:smoke.structural_similarity:0.32666666666666666->0.38280000000000003:improvement=0.05613333333333337
- control_metrics: `{'latency_ms_p50': 28869.28, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.32666666666666666, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 28869.28, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.32666666666666666, 'smoke.binder_reference_f1': 0.0}`
- candidate_metrics: `{'latency_ms_p50': 22892.78, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.38280000000000003, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 22892.78, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.38280000000000003, 'smoke.binder_reference_f1': 0.0}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
