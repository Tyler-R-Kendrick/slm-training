# Continuous cycle `continuous-loop-20260902-int1-verify-02505e13-c1`

- loop_id: `int1-verify`
- cycle_index: `1`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260902-int1-verify-02505e13-c1-control:missing_scoreboard, measurement_incomplete:c20260902-int1-verify-02505e13-c1-semantic-contrast-compiler-margin:missing_scoreboard, harness_failure:c20260902-int1-verify-02505e13-c1-semantic-contrast-compiler-margin:experiment_failed, empty_metrics:e038f2bce220491c613a11286f4c33d10cd7125511b68960f7f1458cf6736fa9, harness_failure:c20260902-int1-verify-02505e13-c1-control:experiment_failed, empty_metrics:911b127e86e599e72cc063b94d571add4405f525e549ac759238c81eb794a7ee, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
