# Continuous cycle `continuous-loop-20260828-continuous-openui-local-8c0b60dd-c605`

- loop_id: `continuous-openui-local`
- cycle_index: `605`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: primary_metric_null_or_worse:smoke.eval_nll:control=4.234666892312747 candidate=7.880536537022405 improvement=-3.6458696447096584, screening_quality_secondary:smoke.structural_similarity:control=0.37569583333333334 candidate=0.1363875:recorded_not_verdict
- control_metrics: `{'latency_ms_p50': 17784.75, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.37569583333333334, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': 4.234666892312747, 'smoke.latency_ms_p50': 17784.75, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.37569583333333334, 'smoke.binder_reference_f1': 0.5313492063492063, 'smoke.eval_nll': 4.234666892312747}`
- candidate_metrics: `{'latency_ms_p50': 6531.05, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.1363875, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': 7.880536537022405, 'smoke.latency_ms_p50': 6531.05, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.1363875, 'smoke.binder_reference_f1': 0.5313492063492063, 'smoke.eval_nll': 7.880536537022405}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive
- speculate: keep_rotating_size_matched_arms
- deltas: `{'binder_reference_f1': 0.0, 'eval_nll': 3.6458696447096584, 'latency_ms_p50': -11253.7, 'meaningful_program_rate': 0.0, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.0, 'smoke.eval_nll': 3.6458696447096584, 'smoke.latency_ms_p50': -11253.7, 'smoke.meaningful_program_rate': 0.0, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': -0.23930833333333335, 'structural_similarity': -0.23930833333333335}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Checkpoint recipe

- `outputs/autoresearch/continuous-loop-20260828-continuous-openui-local-8c0b60dd-c605/runs/c20260828-continuous-openui-local-8c0b60dd-c605-control/checkpoints/last.pt` (6,692,828 bytes); twotower, 1,661,698 trainable parameters, 162 steps, 629 records

- `outputs/autoresearch/continuous-loop-20260828-continuous-openui-local-8c0b60dd-c605/runs/c20260828-continuous-openui-local-8c0b60dd-c605-current-rung-data-heal/checkpoints/last.pt` (6,453,212 bytes); twotower, 1,601,794 trainable parameters, 162 steps, 90 records
