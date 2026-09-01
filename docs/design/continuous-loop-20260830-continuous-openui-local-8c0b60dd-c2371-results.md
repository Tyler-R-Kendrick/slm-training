# Continuous cycle `continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2371`

- loop_id: `continuous-openui-local`
- cycle_index: `2371`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260830-continuous-openui-local-8c0b60dd-c2371-control, fixture_insufficient_n:c20260830-continuous-openui-local-8c0b60dd-c2371-current-rung-data-heal, primary_metric_null_or_worse:smoke.eval_nll:control=4.623752577150769 candidate=9.433029869592197 improvement=-4.809277292441428, screening_quality_secondary:smoke.structural_similarity:control=0.43296666666666667 candidate=0.13118333333333335:recorded_not_verdict, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 13070.11, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.43296666666666667, 'binder_reference_f1': 0.7984126984126984, 'eval_nll': 4.623752577150769, 'smoke.latency_ms_p50': 13070.11, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.43296666666666667, 'smoke.binder_reference_f1': 0.7984126984126984, 'smoke.eval_nll': 4.623752577150769}`
- candidate_metrics: `{'latency_ms_p50': 3110.42, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.13118333333333335, 'binder_reference_f1': 0.5333333333333333, 'eval_nll': 9.433029869592197, 'smoke.latency_ms_p50': 3110.42, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.13118333333333335, 'smoke.binder_reference_f1': 0.5333333333333333, 'smoke.eval_nll': 9.433029869592197}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': -0.265079365079365, 'eval_nll': 4.809277292441428, 'latency_ms_p50': -9959.69, 'meaningful_program_rate': -0.16666666666666666, 'parse_rate': 0.0, 'smoke.binder_reference_f1': -0.265079365079365, 'smoke.eval_nll': 4.809277292441428, 'smoke.latency_ms_p50': -9959.69, 'smoke.meaningful_program_rate': -0.16666666666666666, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': -0.3017833333333333, 'structural_similarity': -0.3017833333333333}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Checkpoint recipe

- `/home/codex/.herdr/worktrees/slm-training/worktree-calm-meadow-8469/outputs/autoresearch/continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2371/runs/c20260830-continuous-openui-local-8c0b60dd-c2371-control/checkpoints/last.pt` (6,692,828 bytes); twotower, 1,661,698 trainable parameters, 94 steps, 629 records

- `/home/codex/.herdr/worktrees/slm-training/worktree-calm-meadow-8469/outputs/autoresearch/continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2371/runs/c20260830-continuous-openui-local-8c0b60dd-c2371-current-rung-data-heal/checkpoints/last.pt` (6,453,212 bytes); twotower, 1,601,794 trainable parameters, 94 steps, 90 records
