# Continuous cycle `continuous-loop-20260828-continuous-openui-local-8c0b60dd-c600`

- loop_id: `continuous-openui-local`
- cycle_index: `600`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260828-continuous-openui-local-8c0b60dd-c600-control:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': 9763.88, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.11787500000000001, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': 9.153625249767549, 'smoke.latency_ms_p50': 9763.88, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.11787500000000001, 'smoke.binder_reference_f1': 0.5313492063492063, 'smoke.eval_nll': 9.153625249767549}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Checkpoint recipe

- `outputs/autoresearch/continuous-loop-20260828-continuous-openui-local-8c0b60dd-c600/runs/c20260828-continuous-openui-local-8c0b60dd-c600-current-rung-data-heal/checkpoints/last.pt` (6,453,212 bytes); twotower, 1,601,794 trainable parameters, 138 steps, 90 records
