# Continuous cycle `continuous-loop-20260826-continuous-openui-local-8c0b60dd-c582`

- loop_id: `continuous-openui-local`
- cycle_index: `582`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260826-continuous-openui-local-8c0b60dd-c582-control:missing_scoreboard, fixture_insufficient_n:c20260826-continuous-openui-local-8c0b60dd-c582-current-rung-data-heal-latprobe, fixture_insufficient_n:c20260826-continuous-openui-local-8c0b60dd-c582-control-latprobe, primary_metric_unavailable, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': 27556.54, 'parse_rate': 1.0, 'meaningful_program_rate': 0.375, 'structural_similarity': 0.22297916666666664, 'binder_reference_f1': 0.9465277777777779, 'eval_nll': None, 'smoke.latency_ms_p50': 27556.54, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.375, 'smoke.structural_similarity': 0.22297916666666664, 'smoke.binder_reference_f1': 0.9465277777777779}`

## Recipe and evidence

- device/backend: CPU / scratch TwoTower
- training: no c582 training; exact frozen replay reused both c581 checkpoints (candidate: 1,601,794 trainable parameters)
- evaluation: smoke suite, `n=24`, seed `100581`, 70-second per-arm wall cap
- decode policy: grammar-constrained, finalize-validated, fail closed; unconstrained fallback disabled
- AgentV: candidate bundle completed; control timed out before a scoreboard
- honesty: fixture screening only, not a ship or promotion claim

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
