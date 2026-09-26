# Continuous cycle `science-lab-supervised-release-r16-20260923`

- loop_id: `science-lab-supervised-release-r16-20260923`
- cycle_index: `1`
- role/intent: `screening` / `diagnostic_remeasurement`
- primary_metric: `smoke.eval_nll`
- positive: **True**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:candidate, fixture_insufficient_n:control, primary_metric_win:smoke.eval_nll:24.23453077471276->22.469210517149463:improvement=1.765320257563296:paired_win:p=0.03125:alpha=1/20:n_pairs=6:n_nontied=6:median_delta=1.740485119401363:minimum_effect=0.02, screening_quality_secondary:smoke.structural_similarity:control=0.24445000000000003 candidate=0.24445000000000003:recorded_not_verdict, fixture_insufficient_n:quality_probe
- control_metrics: `{'latency_ms_p50': 684.43, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.24445000000000003, 'binder_reference_f1': 0.9444444444444445, 'eval_nll': 24.23453077471276, 'smoke.latency_ms_p50': 684.43, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.24445000000000003, 'smoke.binder_reference_f1': 0.9444444444444445, 'smoke.eval_nll': 24.23453077471276}`
- candidate_metrics: `{'latency_ms_p50': 646.14, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.24445000000000003, 'binder_reference_f1': 1.0, 'eval_nll': 22.469210517149463, 'smoke.latency_ms_p50': 646.14, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.24445000000000003, 'smoke.binder_reference_f1': 1.0, 'smoke.eval_nll': 22.469210517149463}`

## Hill-climb this cycle

- went well: measurement_complete, quality_aware_positive
- went wrong: —
- speculate: —
- deltas: `{'binder_reference_f1': 0.05555555555555547, 'eval_nll': -1.765320257563296, 'latency_ms_p50': -38.289999999999964, 'meaningful_program_rate': 0.0, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.05555555555555547, 'smoke.eval_nll': -1.765320257563296, 'smoke.latency_ms_p50': -38.289999999999964, 'smoke.meaningful_program_rate': 0.0, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': 0.0, 'structural_similarity': 0.0}`

## Recipe and interpretation

- CPU diagnostic; seed `7301`; six public smoke cases from six declared root families.
- No training or checkpoint publication occurred. Both size-matched arms reused retained 65,826-parameter checkpoints.
- Ordinary supervisor execution resumed its durable cursors across bounded invocations; both arms produced six complete decoded records, scoreboards, loss reports, and AgentEvals/AgentV evidence.
- Conditional masked-token NLL fell from `24.234531` to `22.469211` (six non-tied pairs, exact two-sided sign-test `p=0.03125`). Parse rate was `1.0` for both arms; meaningful-program rate was `1/6` for both.
- Fixture-volume and quality-probe gates failed for insufficient sample size. This is diagnostic evidence only: no independent confirmation, promotion, shipment, or model-card update.

The immutable execution source digest was `72633465fdab1f2ac2bdcae3b598fd54eb51bae4510c465f15a1ff02c6b1580b`; its version stamp names base commit `e56115002dd3f2872d2d24d1a5ef85ff36ce49cf` and `code_dirty=true`. The result validates that captured source snapshot, not a merged commit. The earlier R14 environment failure and R15 grant-exhaustion attempt remain incomplete records; R16 completed using the existing locked continuation grant. Startup also recorded a readiness-repair wait because no exact `locked_data_readiness_context` was available; no data rebuild was claimed.

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
