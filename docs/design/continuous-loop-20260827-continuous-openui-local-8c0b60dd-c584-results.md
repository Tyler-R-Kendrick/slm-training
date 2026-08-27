# Continuous cycle `continuous-loop-20260827-continuous-openui-local-8c0b60dd-c584`

- loop_id: `continuous-openui-local`
- cycle_index: `584`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260827-continuous-openui-local-8c0b60dd-c584-control:missing_scoreboard, fixture_insufficient_n:c20260827-continuous-openui-local-8c0b60dd-c584-control-latprobe, fixture_insufficient_n:c20260827-continuous-openui-local-8c0b60dd-c584-current-rung-data-heal-latprobe, primary_metric_unavailable, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': 6741.64, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.1363875, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': None, 'smoke.latency_ms_p50': 6741.64, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.1363875, 'smoke.binder_reference_f1': 0.5313492063492063}`

## Recipe and evidence

- device/backend: CPU / scratch TwoTower
- training: no c584 training; exact frozen replay reused both c583 checkpoints (candidate: 1,601,794 trainable parameters)
- evaluation: smoke suite, `n=24`, seed `100583`, 70-second per-arm wall cap
- decode policy: grammar-constrained, finalize-validated, fail closed; unconstrained fallback disabled
- AgentV: candidate bundle completed; control timed out before a scoreboard
- honesty: fixture screening only, not a ship or promotion claim

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
