# Continuous cycle `continuous-loop-20260826-continuous-openui-local-8c0b60dd-c567`

- loop_id: `continuous-openui-local`
- cycle_index: `567`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260826-continuous-openui-local-8c0b60dd-c567-control:missing_scoreboard, measurement_incomplete:c20260826-continuous-openui-local-8c0b60dd-c567-current-rung-data-heal:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Harness evidence

- The frozen manifests and c565 training checkpoints were hash-verified.
- Training reuse failed closed because the plan contained two evaluation stages (latency probe plus full suite), while the reuse seam accepted exactly one.
- Both 70-second arms therefore reran training and timed out before scoreboards; c567 created no checkpoint that supersedes c565.
- v290 binds the verified checkpoint into every declared evaluation stage. No recipe, metric, or gate is reused.
