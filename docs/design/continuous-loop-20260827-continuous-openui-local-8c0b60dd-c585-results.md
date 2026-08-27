# Continuous cycle `continuous-loop-20260827-continuous-openui-local-8c0b60dd-c585`

- loop_id: `continuous-openui-local`
- cycle_index: `585`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260827-continuous-openui-local-8c0b60dd-c585-control:missing_scoreboard, measurement_incomplete:c20260827-continuous-openui-local-8c0b60dd-c585-current-rung-data-heal:smoke:incomplete_document_n=24:decode_timeout_count=24, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': 9.054562708409032, 'smoke.eval_nll': 9.054562708409032}`

## Recipe and evidence

- control: CPU scratch TwoTower, 1,661,698 trainable parameters, 400 steps, seed `100585`, 629 records, 42.28 s train, final loss 6.81211
- candidate: CPU scratch TwoTower, 1,601,794 trainable parameters, 400 steps, seed `100585`, 90 records, 16.52 s train, final loss 0.000869098
- evaluation: smoke suite, `n=24`, 70-second per-arm wall cap; all 24 candidate documents were incomplete from decode timeouts
- decode policy: grammar-constrained and fail closed; unconstrained fallback disabled
- checkpoints: local scratch only; no upload or promotion
- honesty: candidate eval NLL 9.0546 is retained diagnostically, but quality metrics and the matched comparison are unavailable

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, fit_decode_timeout_to_n_times_p50
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
