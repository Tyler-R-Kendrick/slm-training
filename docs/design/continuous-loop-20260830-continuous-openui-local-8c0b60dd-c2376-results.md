# Continuous cycle `continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2376`

- loop_id: `continuous-openui-local`
- cycle_index: `2376`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260830-continuous-openui-local-8c0b60dd-c2376-control:missing_scoreboard, fixture_insufficient_n:c20260830-continuous-openui-local-8c0b60dd-c2376-current-rung-data-heal, primary_metric_unavailable, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': 4755.71, 'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'structural_similarity': 0.2449, 'binder_reference_f1': 0.5333333333333333, 'eval_nll': None, 'smoke.latency_ms_p50': 4755.71, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.3333333333333333, 'smoke.structural_similarity': 0.2449, 'smoke.binder_reference_f1': 0.5333333333333333}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Checkpoint recipe

- `/home/codex/.herdr/worktrees/slm-training/worktree-calm-meadow-8469/outputs/autoresearch/continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2376/runs/c20260830-continuous-openui-local-8c0b60dd-c2376-control/checkpoints/last.pt` (6,692,828 bytes); twotower, 1,661,698 trainable parameters, 128 steps, 629 records

- `/home/codex/.herdr/worktrees/slm-training/worktree-calm-meadow-8469/outputs/autoresearch/continuous-loop-20260830-continuous-openui-local-8c0b60dd-c2376/runs/c20260830-continuous-openui-local-8c0b60dd-c2376-current-rung-data-heal/checkpoints/last.pt` (6,453,212 bytes); twotower, 1,601,794 trainable parameters, 128 steps, 90 records
