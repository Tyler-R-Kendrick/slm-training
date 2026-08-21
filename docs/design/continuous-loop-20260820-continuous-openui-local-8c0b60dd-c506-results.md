# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c506`

- loop_id: `continuous-openui-local`
- cycle_index: `506`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c506-current-rung-data-heal, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c506-control, primary_metric_null_or_worse:smoke.structural_similarity:control=0.13118333333333335 candidate=0.13118333333333335 improvement=0.0, fixture_insufficient_n_alone, mechanism_no_effect:quality_metrics_identical, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 3123.47, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.13118333333333335, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 3123.47, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.13118333333333335, 'smoke.binder_reference_f1': 0.5333333333333333}`
- candidate_metrics: `{'latency_ms_p50': 4283.05, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.13118333333333335, 'binder_reference_f1': 0.5333333333333333, 'smoke.latency_ms_p50': 4283.05, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.13118333333333335, 'smoke.binder_reference_f1': 0.5333333333333333}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, mechanism_no_effect:quality_metrics_identical, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm, retire_knob_do_not_confirm
- deltas: `{'binder_reference_f1': 0.0, 'latency_ms_p50': 1159.5800000000004, 'meaningful_program_rate': 0.0, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.0, 'smoke.latency_ms_p50': 1159.5800000000004, 'smoke.meaningful_program_rate': 0.0, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': 0.0, 'structural_similarity': 0.0}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
