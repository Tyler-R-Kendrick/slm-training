# Continuous cycle `continuous-loop-20260826-continuous-openui-local-8c0b60dd-c570`

- loop_id: `continuous-openui-local`
- cycle_index: `570`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- recipe: local CPU, scratch context, 173 steps, batch size 2, grammar-constrained, two 70-second arm walls
- reasons: measurement_incomplete:c20260826-continuous-openui-local-8c0b60dd-c570-control:smoke:incomplete_document_n=24:decode_timeout_count=24, primary_metric_null_or_worse:smoke.eval_nll:control=4.417992569098141 candidate=8.454308676348 improvement=-4.03631610724986, screening_quality_secondary:smoke.structural_similarity:control=None candidate=0.1363875:recorded_not_verdict
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': 4.417992569098141, 'smoke.eval_nll': 4.417992569098141}`
- candidate_metrics: `{'latency_ms_p50': 6899.85, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.1363875, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': 8.454308676348, 'smoke.latency_ms_p50': 6899.85, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.1363875, 'smoke.binder_reference_f1': 0.5313492063492063, 'smoke.eval_nll': 8.454308676348}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, fit_decode_timeout_to_n_times_p50
- deltas: `{'eval_nll': 4.03631610724986, 'smoke.eval_nll': 4.03631610724986}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

The exact-head command-import repair reduced autoresearch startup to 0.27s and
cleared the pre-arm symmetric-budget guard without changing either arm. The
candidate completed all 24 smoke documents but failed quality gates. The
control produced a checkpoint and loss evidence but timed out all 24 document
decodes; therefore the paired campaign remains incomplete and non-promotable.
