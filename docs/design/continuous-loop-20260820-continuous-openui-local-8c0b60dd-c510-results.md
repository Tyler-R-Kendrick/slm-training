# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c510`

- loop_id: `continuous-openui-local`
- cycle_index: `510`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260820-continuous-openui-local-8c0b60dd-c510-control:missing_scoreboard, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c510-current-rung-data-heal, primary_metric_unavailable, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None}`
- candidate_metrics: `{'latency_ms_p50': 4481.46, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.16018333333333332, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 4481.46, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.16018333333333332, 'smoke.binder_reference_f1': 0.5333333333333333}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
