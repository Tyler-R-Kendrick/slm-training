# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c520`

- loop_id: `continuous-openui-local`
- cycle_index: `520`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c520-current-rung-data-heal, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c520-control, primary_metric_win:smoke.structural_similarity:0.13118333333333335->0.3146166666666667:improvement=0.18343333333333336, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 4441.56, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.13118333333333335, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 4441.56, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.13118333333333335, 'smoke.binder_reference_f1': 0.5333333333333333}`
- candidate_metrics: `{'latency_ms_p50': 7465.88, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.3146166666666667, 'binder_reference_f1': 0.7666666666666666, 'smoke.latency_ms_p50': 7465.88, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.3146166666666667, 'smoke.binder_reference_f1': 0.7666666666666666}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': 0.23333333333333328, 'latency_ms_p50': 3024.3199999999997, 'meaningful_program_rate': -0.16666666666666666, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.23333333333333328, 'smoke.latency_ms_p50': 3024.3199999999997, 'smoke.meaningful_program_rate': -0.16666666666666666, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': 0.18343333333333336, 'structural_similarity': 0.18343333333333336}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
