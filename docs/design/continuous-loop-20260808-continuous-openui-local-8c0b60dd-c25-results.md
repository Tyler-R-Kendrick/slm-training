# Continuous cycle `continuous-loop-20260808-continuous-openui-local-8c0b60dd-c25`

- loop_id: `continuous-openui-local`
- cycle_index: `25`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.structural_similarity`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260808-continuous-openui-local-8c0b60dd-c25-binder-component-plan, fixture_insufficient_n:c20260808-continuous-openui-local-8c0b60dd-c25-control, primary_metric_null_or_worse:smoke.structural_similarity:control=0.3764 candidate=0.3764 improvement=0.0, fixture_insufficient_n_alone
- control_metrics: `{'latency_ms_p50': 14337.96, 'parse_rate': 1.0, 'meaningful_program_rate': 0.6666666666666666, 'structural_similarity': 0.3764, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 14337.96, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.6666666666666666, 'smoke.structural_similarity': 0.3764, 'smoke.binder_reference_f1': 0.9523809523809524}`
- candidate_metrics: `{'latency_ms_p50': 14876.65, 'parse_rate': 1.0, 'meaningful_program_rate': 0.6666666666666666, 'structural_similarity': 0.3764, 'binder_reference_f1': 0.9523809523809524, 'smoke.latency_ms_p50': 14876.65, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.6666666666666666, 'smoke.structural_similarity': 0.3764, 'smoke.binder_reference_f1': 0.9523809523809524}`

**Reconstruction note:** this cycle's `cycle_handoff.json` was never written by the
continuous driver — `_self_heal_document_actions` requires that file and silently
no-ops without it (`scripts/run_autotrain_continuous.py:2570`), so the c25
campaign's evidence (`campaign.json`, `results.tsv`, `events.jsonl`,
`runs/*/scoreboard.json`) completed and sat undocumented while the driver moved
on to cycle c26 in the next invocation. Values above are read directly from the
committed run artifacts (`runs/c20260808-continuous-openui-local-8c0b60dd-c25-{binder-component-plan,control}/scoreboard.json`)
rather than rendered by `_render_continuous_cycle_docs`. Fixture screening only —
not a ship claim. The missing-handoff gap itself is a harness signal worth a
`repair_harness` follow-up (family: `autotrain`/`experiments`) so ordinary
document closeout can't silently skip a completed cycle again.
