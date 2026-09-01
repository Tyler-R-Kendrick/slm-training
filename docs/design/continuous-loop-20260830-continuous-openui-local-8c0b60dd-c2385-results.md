# Continuous cycle `continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2385`

- loop_id: `continuous-openui-local`
- cycle_index: `2385`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260830-continuous-openui-local-8c0b60dd-c2385-control, fixture_insufficient_n:c20260830-continuous-openui-local-8c0b60dd-c2385-current-rung-data-heal, primary_metric_null_or_worse:smoke.eval_nll:control=4.255415355541185 candidate=8.176482707460293 improvement=-3.921067351919108, screening_quality_secondary:smoke.structural_similarity:control=0.1074 candidate=0.13118333333333335:recorded_not_verdict, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 21603.74, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.1074, 'binder_reference_f1': 1.0, 'eval_nll': 4.255415355541185, 'smoke.latency_ms_p50': 21603.74, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.1074, 'smoke.binder_reference_f1': 1.0, 'smoke.eval_nll': 4.255415355541185}`
- candidate_metrics: `{'latency_ms_p50': 3223.01, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.13118333333333335, 'binder_reference_f1': 0.5333333333333333, 'eval_nll': 8.176482707460293, 'smoke.latency_ms_p50': 3223.01, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.13118333333333335, 'smoke.binder_reference_f1': 0.5333333333333333, 'smoke.eval_nll': 8.176482707460293}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': -0.4666666666666667, 'eval_nll': 3.921067351919108, 'latency_ms_p50': -18380.730000000003, 'meaningful_program_rate': -0.16666666666666666, 'parse_rate': 0.0, 'smoke.binder_reference_f1': -0.4666666666666667, 'smoke.eval_nll': 3.921067351919108, 'smoke.latency_ms_p50': -18380.730000000003, 'smoke.meaningful_program_rate': -0.16666666666666666, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': 0.02378333333333335, 'structural_similarity': 0.02378333333333335}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
