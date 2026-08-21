# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c529`

- loop_id: `continuous-openui-local`
- cycle_index: `529`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c529-current-rung-data-heal, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c529-control, primary_metric_null_or_worse:smoke.structural_similarity:control=0.07673333333333333 candidate=0.0534 improvement=-0.02333333333333333, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 3644.72, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.07673333333333333, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 3644.72, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.07673333333333333, 'smoke.binder_reference_f1': 0.5333333333333333}`
- candidate_metrics: `{'latency_ms_p50': 2741.64, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.0534, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 2741.64, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.0534, 'smoke.binder_reference_f1': 0.5333333333333333}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': 0.0, 'latency_ms_p50': -903.0799999999999, 'meaningful_program_rate': 0.0, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.0, 'smoke.latency_ms_p50': -903.0799999999999, 'smoke.meaningful_program_rate': 0.0, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': -0.02333333333333333, 'structural_similarity': -0.02333333333333333}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
