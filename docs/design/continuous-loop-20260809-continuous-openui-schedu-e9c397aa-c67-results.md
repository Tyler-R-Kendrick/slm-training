# Continuous cycle `continuous-loop-20260809-continuous-openui-schedu-e9c397aa-c67`

- loop_id: `continuous-openui-scheduled-2805`
- cycle_index: `67`
- role/intent: `screening` / `confirm`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260809-continuous-openui-schedu-e9c397aa-c67-confirm, fixture_insufficient_n:c20260809-continuous-openui-schedu-e9c397aa-c67-control, primary_quality_win_rejected_latency_budget:smoke.structural_similarity:lat=3708.27->9133.42, quality_win_rejected_latency_budget:mpr=0.0->0.6666666666666666 lat=3708.27->9133.42, primary_metric_win:smoke.structural_similarity:0.057499999999999996->0.3086:improvement=0.2511, confirmation_rejected:primary_quality_not_reheld
- control_metrics: `{'latency_ms_p50': 3708.27, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.057499999999999996, 'binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 3708.27, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.057499999999999996, 'smoke.binder_reference_f1': 0.0}`
- candidate_metrics: `{'latency_ms_p50': 9133.42, 'parse_rate': 1.0, 'meaningful_program_rate': 0.6666666666666666, 'structural_similarity': 0.3086, 'binder_reference_f1': 0.6333333333333333, 'smoke.latency_ms_p50': 9133.42, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.6666666666666666, 'smoke.structural_similarity': 0.3086, 'smoke.binder_reference_f1': 0.6333333333333333}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
