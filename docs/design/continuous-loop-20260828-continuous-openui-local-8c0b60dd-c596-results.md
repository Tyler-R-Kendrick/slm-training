# Continuous cycle `continuous-loop-20260828-continuous-openui-local-8c0b60dd-c596`

- loop_id: `continuous-openui-local`
- cycle_index: `596`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260828-continuous-openui-local-8c0b60dd-c596-control:smoke:incomplete_document_n=16:decode_timeout_count=16, measurement_incomplete:c20260828-continuous-openui-local-8c0b60dd-c596-current-rung-data-heal:smoke:incomplete_document_n=24:decode_timeout_count=24, primary_metric_null_or_worse:smoke.eval_nll:control=4.698059719685613 candidate=7.867653502040523 improvement=-3.1695937823549096, screening_quality_secondary:smoke.structural_similarity:control=0.2379375 candidate=None:recorded_not_verdict
- control_metrics: `{'latency_ms_p50': 11764.15, 'parse_rate': 1.0, 'meaningful_program_rate': 0.25, 'structural_similarity': 0.2379375, 'binder_reference_f1': 0.90625, 'eval_nll': 4.698059719685613, 'smoke.latency_ms_p50': 11764.15, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.25, 'smoke.structural_similarity': 0.2379375, 'smoke.binder_reference_f1': 0.90625, 'smoke.eval_nll': 4.698059719685613}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': 7.867653502040523, 'smoke.eval_nll': 7.867653502040523}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, fit_decode_timeout_to_n_times_p50
- deltas: `{'eval_nll': 3.1695937823549096, 'smoke.eval_nll': 3.1695937823549096}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Checkpoint recipe

- `outputs/autoresearch/continuous-loop-20260828-continuous-openui-local-8c0b60dd-c596/runs/c20260828-continuous-openui-local-8c0b60dd-c596-control/checkpoints/last.pt` (6,692,828 bytes); twotower, 1,661,698 trainable parameters, 97 steps, 629 records

- `outputs/autoresearch/continuous-loop-20260828-continuous-openui-local-8c0b60dd-c596/runs/c20260828-continuous-openui-local-8c0b60dd-c596-current-rung-data-heal/checkpoints/last.pt` (6,453,212 bytes); twotower, 1,601,794 trainable parameters, 97 steps, 90 records
