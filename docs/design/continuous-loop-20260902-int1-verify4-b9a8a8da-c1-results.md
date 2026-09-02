# Continuous cycle `continuous-loop-20260902-int1-verify4-b9a8a8da-c1`

- loop_id: `int1-verify4`
- cycle_index: `1`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260902-int1-verify4-b9a8a8da-c1-control:missing_scoreboard, measurement_incomplete:c20260902-int1-verify4-b9a8a8da-c1-semantic-contrast-compiler-margin:missing_scoreboard, harness_failure:c20260902-int1-verify4-b9a8a8da-c1-control:experiment_failed, empty_metrics:241daadcec00b1bf0496cd82c376b858019c69e07c4c2fca170fa6037e8512da, harness_failure:c20260902-int1-verify4-b9a8a8da-c1-semantic-contrast-compiler-margin:experiment_failed, empty_metrics:e15a42aa0def27b1f81379e84e8027eb886f9e4b1e5b65eeedddf12bfb635699, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
