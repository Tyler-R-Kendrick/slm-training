# Continuous cycle `continuous-loop-20260826-continuous-openui-local-8c0b60dd-c583`

- loop_id: `continuous-openui-local`
- cycle_index: `583`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260826-continuous-openui-local-8c0b60dd-c583-control:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': 6685.78, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.1363875, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': 8.354336838291324, 'smoke.latency_ms_p50': 6685.78, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.1363875, 'smoke.binder_reference_f1': 0.5313492063492063, 'smoke.eval_nll': 8.354336838291324}`

## Recipe and evidence

- control: CPU scratch TwoTower, 1,661,698 trainable parameters, 401 steps, seed `100583`, 629 records, 40.68 s train, final loss 6.92503; timed out before a scoreboard
- candidate: CPU scratch TwoTower, 1,601,794 trainable parameters, 401 steps, seed `100583`, 90 records, 16.48 s train, final loss 0.00275301
- evaluation: smoke suite, `n=24`, 70-second per-arm wall cap; candidate AgentV bundle completed with 4/9 assertions passing
- decode policy: grammar-constrained and fail closed; unconstrained fallback disabled
- checkpoints: local scratch only; no upload or promotion
- honesty: the missing control scoreboard makes this comparison incomplete and non-positive

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
