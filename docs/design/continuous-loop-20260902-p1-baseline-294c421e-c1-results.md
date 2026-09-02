# Continuous cycle `continuous-loop-20260902-p1-baseline-294c421e-c1`

- loop_id: `p1-baseline`
- cycle_index: `1`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260902-p1-baseline-294c421e-c1-control:missing_scoreboard, measurement_incomplete:c20260902-p1-baseline-294c421e-c1-semantic-contrast-compiler-margin:missing_scoreboard, harness_failure:c20260902-p1-baseline-294c421e-c1-semantic-contrast-compiler-margin:experiment_failed, empty_metrics:cfe93de3933159791df5e1b238ded18060cf0239c6238041fea4b23e6cf70216, harness_failure:c20260902-p1-baseline-294c421e-c1-control:experiment_failed, empty_metrics:f54b0b53a628c43e4f53db17bcd5dda6710085811916141af47686984a891424, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
