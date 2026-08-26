# Continuous cycle `continuous-loop-20260821-continuous-openui-local-8c0b60dd-c550`

- loop_id: `continuous-openui-local`
- cycle_index: `550`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260821-continuous-openui-local-8c0b60dd-c550-control:missing_scoreboard, measurement_incomplete:c20260821-continuous-openui-local-8c0b60dd-c550-current-rung-data-heal:missing_scoreboard, harness_failure:c20260821-continuous-openui-local-8c0b60dd-c550-control:experiment_failed, empty_metrics:f4091a8fabd94ebf1310bcc6f6865c23f7c5a145179b5bb96a62f638d88b075c, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
