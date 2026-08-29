# Continuous cycle `continuous-loop-20260828-continuous-openui-local-8c0b60dd-c616`

- loop_id: `continuous-openui-local`
- cycle_index: `616`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260828-continuous-openui-local-8c0b60dd-c616-control:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': 17443.14, 'parse_rate': 1.0, 'meaningful_program_rate': 0.08333333333333333, 'structural_similarity': 0.07894583333333334, 'binder_reference_f1': 0.6791666666666667, 'eval_nll': 7.535711227059323, 'smoke.latency_ms_p50': 17443.14, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.08333333333333333, 'smoke.structural_similarity': 0.07894583333333334, 'smoke.binder_reference_f1': 0.6791666666666667, 'smoke.eval_nll': 7.535711227059323}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Checkpoint recipe

- `outputs/autoresearch/continuous-loop-20260828-continuous-openui-local-8c0b60dd-c616/runs/c20260828-continuous-openui-local-8c0b60dd-c616-current-rung-data-heal/checkpoints/last.pt` (6,453,212 bytes); twotower, 1,601,794 trainable parameters, 161 steps, 90 records
