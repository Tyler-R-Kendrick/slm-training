# Continuous cycle `continuous-loop-20260828-continuous-openui-local-8c0b60dd-c606`

- loop_id: `continuous-openui-local`
- cycle_index: `606`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: primary_metric_null_or_worse:smoke.eval_nll:control=3.6768596104926634 candidate=9.205169427808679 improvement=-5.528309817316016, screening_quality_secondary:smoke.structural_similarity:control=0.37569583333333334 candidate=0.11787500000000001:recorded_not_verdict
- control_metrics: `{'latency_ms_p50': 15003.5, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.37569583333333334, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': 3.6768596104926634, 'smoke.latency_ms_p50': 15003.5, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.37569583333333334, 'smoke.binder_reference_f1': 0.5313492063492063, 'smoke.eval_nll': 3.6768596104926634}`
- candidate_metrics: `{'latency_ms_p50': 6185.19, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.11787500000000001, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': 9.205169427808679, 'smoke.latency_ms_p50': 6185.19, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.11787500000000001, 'smoke.binder_reference_f1': 0.5313492063492063, 'smoke.eval_nll': 9.205169427808679}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive
- speculate: keep_rotating_size_matched_arms
- deltas: `{'binder_reference_f1': 0.0, 'eval_nll': 5.528309817316016, 'latency_ms_p50': -8818.310000000001, 'meaningful_program_rate': 0.0, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.0, 'smoke.eval_nll': 5.528309817316016, 'smoke.latency_ms_p50': -8818.310000000001, 'smoke.meaningful_program_rate': 0.0, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': -0.25782083333333333, 'structural_similarity': -0.25782083333333333}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Checkpoint recipe

- `outputs/autoresearch/continuous-loop-20260828-continuous-openui-local-8c0b60dd-c606/runs/c20260828-continuous-openui-local-8c0b60dd-c606-control/checkpoints/last.pt` (6,692,828 bytes); twotower, 1,661,698 trainable parameters, 177 steps, 629 records

- `outputs/autoresearch/continuous-loop-20260828-continuous-openui-local-8c0b60dd-c606/runs/c20260828-continuous-openui-local-8c0b60dd-c606-current-rung-data-heal/checkpoints/last.pt` (6,453,212 bytes); twotower, 1,601,794 trainable parameters, 177 steps, 90 records
