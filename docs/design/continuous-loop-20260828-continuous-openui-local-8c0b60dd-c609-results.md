# Continuous cycle `continuous-loop-20260828-continuous-openui-local-8c0b60dd-c609`

- loop_id: `continuous-openui-local`
- cycle_index: `609`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260828-continuous-openui-local-8c0b60dd-c609-control:smoke:incomplete_document_n=24:decode_timeout_count=24, primary_metric_null_or_worse:smoke.eval_nll:control=3.730566986370322 candidate=8.531175070399586 improvement=-4.800608084029264, screening_quality_secondary:smoke.structural_similarity:control=None candidate=0.1363875:recorded_not_verdict
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': 3.730566986370322, 'smoke.eval_nll': 3.730566986370322}`
- candidate_metrics: `{'latency_ms_p50': 6853.38, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.1363875, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': 8.531175070399586, 'smoke.latency_ms_p50': 6853.38, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.1363875, 'smoke.binder_reference_f1': 0.5313492063492063, 'smoke.eval_nll': 8.531175070399586}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, fit_decode_timeout_to_n_times_p50
- deltas: `{'eval_nll': 4.800608084029264, 'smoke.eval_nll': 4.800608084029264}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Checkpoint recipe

- `outputs/autoresearch/continuous-loop-20260828-continuous-openui-local-8c0b60dd-c609/runs/c20260828-continuous-openui-local-8c0b60dd-c609-control/checkpoints/last.pt` (6,692,828 bytes); twotower, 1,661,698 trainable parameters, 178 steps, 629 records

- `outputs/autoresearch/continuous-loop-20260828-continuous-openui-local-8c0b60dd-c609/runs/c20260828-continuous-openui-local-8c0b60dd-c609-current-rung-data-heal/checkpoints/last.pt` (6,453,212 bytes); twotower, 1,601,794 trainable parameters, 178 steps, 90 records
