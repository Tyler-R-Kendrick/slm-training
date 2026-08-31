# Continuous cycle `continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2372`

- loop_id: `continuous-openui-local`
- cycle_index: `2372`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260830-continuous-openui-local-8c0b60dd-c2372-current-rung-data-heal, fixture_insufficient_n:c20260830-continuous-openui-local-8c0b60dd-c2372-control, primary_metric_null_or_worse:smoke.eval_nll:control=4.146907375535855 candidate=7.334767020401859 improvement=-3.1878596448660037, screening_quality_secondary:smoke.structural_similarity:control=0.39349999999999996 candidate=0.13118333333333335:recorded_not_verdict, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 5768.78, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.39349999999999996, 'binder_reference_f1': 0.5333333333333333, 'eval_nll': 4.146907375535855, 'smoke.latency_ms_p50': 5768.78, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.39349999999999996, 'smoke.binder_reference_f1': 0.5333333333333333, 'smoke.eval_nll': 4.146907375535855}`
- candidate_metrics: `{'latency_ms_p50': 3031.95, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.13118333333333335, 'binder_reference_f1': 0.5333333333333333, 'eval_nll': 7.334767020401859, 'smoke.latency_ms_p50': 3031.95, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.13118333333333335, 'smoke.binder_reference_f1': 0.5333333333333333, 'smoke.eval_nll': 7.334767020401859}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': 0.0, 'eval_nll': 3.1878596448660037, 'latency_ms_p50': -2736.83, 'meaningful_program_rate': 0.0, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.0, 'smoke.eval_nll': 3.1878596448660037, 'smoke.latency_ms_p50': -2736.83, 'smoke.meaningful_program_rate': 0.0, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': -0.26231666666666664, 'structural_similarity': -0.26231666666666664}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Checkpoint recipe

- `/home/codex/.herdr/worktrees/slm-training/worktree-calm-meadow-8469/outputs/autoresearch/continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2372/runs/c20260830-continuous-openui-local-8c0b60dd-c2372-control/checkpoints/last.pt` (6,692,828 bytes); twotower, 1,661,698 trainable parameters, 150 steps, 629 records

- `/home/codex/.herdr/worktrees/slm-training/worktree-calm-meadow-8469/outputs/autoresearch/continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2372/runs/c20260830-continuous-openui-local-8c0b60dd-c2372-current-rung-data-heal/checkpoints/last.pt` (6,453,212 bytes); twotower, 1,601,794 trainable parameters, 150 steps, 90 records
