# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c525`

- loop_id: `continuous-openui-local`
- cycle_index: `525`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c525-control, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c525-current-rung-data-heal, non_regression_fail:binder_reference_f1:0.7666666666666666->0.5333333333333333, primary_metric_null_or_worse:smoke.structural_similarity:control=0.0534 candidate=0.2826166666666667 improvement=0.22921666666666668, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 3636.69, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.0534, 'binder_reference_f1': 0.7666666666666666, 'smoke.latency_ms_p50': 3636.69, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.0534, 'smoke.binder_reference_f1': 0.7666666666666666}`
- candidate_metrics: `{'latency_ms_p50': 12982.8, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.2826166666666667, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 12982.8, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.2826166666666667, 'smoke.binder_reference_f1': 0.5333333333333333}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': -0.23333333333333328, 'latency_ms_p50': 9346.109999999999, 'meaningful_program_rate': 0.0, 'parse_rate': 0.0, 'smoke.binder_reference_f1': -0.23333333333333328, 'smoke.latency_ms_p50': 9346.109999999999, 'smoke.meaningful_program_rate': 0.0, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': 0.22921666666666668, 'structural_similarity': 0.22921666666666668}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
