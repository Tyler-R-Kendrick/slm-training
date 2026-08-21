# Continuous cycle `continuous-loop-20260820-continuous-openui-local-8c0b60dd-c464`

- loop_id: `continuous-openui-local`
- cycle_index: `464`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c464-current-rung-data-heal, fixture_insufficient_n:c20260820-continuous-openui-local-8c0b60dd-c464-control, quality_win_rejected_latency_budget:mpr=0.0->0.3333333333333333 lat=5291.4->7522.92, non_regression_fail:binder_reference_f1:0.8222222222222223->0.5, primary_metric_null_or_worse:smoke.structural_similarity:control=0.1725 candidate=0.23803333333333332 improvement=0.06553333333333333, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': 5291.4, 'parse_rate': 1.0, 'meaningful_program_rate': 0.0, 'structural_similarity': 0.1725, 'binder_reference_f1': 0.8222222222222223, 'smoke.latency_ms_p50': 5291.4, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.0, 'smoke.structural_similarity': 0.1725, 'smoke.binder_reference_f1': 0.8222222222222223}`
- candidate_metrics: `{'latency_ms_p50': 7522.92, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.23803333333333332, 'binder_reference_f1': 0.5, 'smoke.latency_ms_p50': 7522.92, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.23803333333333332, 'smoke.binder_reference_f1': 0.5}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
