# Continuous cycle `continuous-loop-20260827-continuous-openui-local-8c0b60dd-c587`

- loop_id: `continuous-openui-local`
- cycle_index: `587`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260827-continuous-openui-local-8c0b60dd-c587-control:missing_scoreboard, measurement_incomplete:c20260827-continuous-openui-local-8c0b60dd-c587-current-rung-data-heal:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`

## Checkpoint recipes

- control: local scratch TwoTower checkpoint at `outputs/autoresearch/continuous-loop-20260827-continuous-openui-local-8c0b60dd-c587/runs/c20260827-continuous-openui-local-8c0b60dd-c587-control/checkpoints/last.pt`; 1,661,698 trainable parameters, CPU, 173 steps, seed 100587, 629 records, 19.95 s, final loss 8.51234, 6,692,828 bytes.
- candidate: local scratch TwoTower checkpoint at `outputs/autoresearch/continuous-loop-20260827-continuous-openui-local-8c0b60dd-c587/runs/c20260827-continuous-openui-local-8c0b60dd-c587-current-rung-data-heal/checkpoints/last.pt`; 1,601,794 trainable parameters, CPU, 173 steps, seed 100587, 90 records, 8.23 s, final loss 2.29963, 6,453,212 bytes.
- checkpoint sync: disabled for this fixture/scratch screening cycle.
- evaluation: both arms hit the 70 s cap before scoreboards or eval NLL. No quality comparison or promotion claim is available.

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
