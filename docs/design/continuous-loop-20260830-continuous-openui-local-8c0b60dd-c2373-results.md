# Continuous cycle `continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2373`

- loop_id: `continuous-openui-local`
- cycle_index: `2373`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `True`
- evidence_class: `fixture`
- reasons: fixture_insufficient_n:c20260830-continuous-openui-local-8c0b60dd-c2373-control, fixture_insufficient_n:c20260830-continuous-openui-local-8c0b60dd-c2373-current-rung-data-heal, primary_metric_null_or_worse:smoke.eval_nll:control=4.4364331289500205 candidate=8.158025523987348 improvement=-3.7215923950373275, screening_quality_secondary:smoke.structural_similarity:control=0.3763 candidate=0.13118333333333335:recorded_not_verdict, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': 8218.74, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.3763, 'binder_reference_f1': 0.5333333333333333, 'eval_nll': 4.4364331289500205, 'smoke.latency_ms_p50': 8218.74, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.3763, 'smoke.binder_reference_f1': 0.5333333333333333, 'smoke.eval_nll': 4.4364331289500205}`
- candidate_metrics: `{'latency_ms_p50': 6533.79, 'parse_rate': 1.0, 'meaningful_program_rate': 0.16666666666666666, 'structural_similarity': 0.13118333333333335, 'binder_reference_f1': 0.5333333333333333, 'eval_nll': 8.158025523987348, 'smoke.latency_ms_p50': 6533.79, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.16666666666666666, 'smoke.structural_similarity': 0.13118333333333335, 'smoke.binder_reference_f1': 0.5333333333333333, 'smoke.eval_nll': 8.158025523987348}`

## Hill-climb this cycle

- went well: measurement_complete
- went wrong: non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{'binder_reference_f1': 0.0, 'eval_nll': 3.7215923950373275, 'latency_ms_p50': -1684.9499999999998, 'meaningful_program_rate': 0.0, 'parse_rate': 0.0, 'smoke.binder_reference_f1': 0.0, 'smoke.eval_nll': 3.7215923950373275, 'smoke.latency_ms_p50': -1684.9499999999998, 'smoke.meaningful_program_rate': 0.0, 'smoke.parse_rate': 0.0, 'smoke.structural_similarity': -0.24511666666666668, 'structural_similarity': -0.24511666666666668}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Checkpoint recipe

- `/home/codex/.herdr/worktrees/slm-training/worktree-calm-meadow-8469/outputs/autoresearch/continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2373/runs/c20260830-continuous-openui-local-8c0b60dd-c2373-control/checkpoints/last.pt` (6,692,828 bytes); twotower, 1,661,698 trainable parameters, 167 steps, 629 records

- `/home/codex/.herdr/worktrees/slm-training/worktree-calm-meadow-8469/outputs/autoresearch/continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2373/runs/c20260830-continuous-openui-local-8c0b60dd-c2373-current-rung-data-heal/checkpoints/last.pt` (6,453,212 bytes); twotower, 1,601,794 trainable parameters, 167 steps, 90 records
