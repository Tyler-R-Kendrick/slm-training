# Continuous cycle `continuous-loop-20260827-continuous-openui-local-8c0b60dd-c586`

- loop_id: `continuous-openui-local`
- cycle_index: `586`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260827-continuous-openui-local-8c0b60dd-c586-control:missing_scoreboard, measurement_incomplete:c20260827-continuous-openui-local-8c0b60dd-c586-current-rung-data-heal:smoke:incomplete_document_n=24:decode_timeout_count=24, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': 7.298664159627289, 'smoke.eval_nll': 7.298664159627289}`

## Checkpoint recipes

- control: local scratch TwoTower checkpoint at `outputs/autoresearch/continuous-loop-20260827-continuous-openui-local-8c0b60dd-c586/runs/c20260827-continuous-openui-local-8c0b60dd-c586-control/checkpoints/last.pt`; 1,661,698 trainable parameters, CPU, 401 steps, seed 100586, 629 records, 42.03 s, final loss 7.23869, 6,692,828 bytes.
- candidate: local scratch TwoTower checkpoint at `outputs/autoresearch/continuous-loop-20260827-continuous-openui-local-8c0b60dd-c586/runs/c20260827-continuous-openui-local-8c0b60dd-c586-current-rung-data-heal/checkpoints/last.pt`; 1,601,794 trainable parameters, CPU, 401 steps, seed 100586, 90 records, 17.15 s, final loss 0.00772793, 6,453,212 bytes.
- checkpoint sync: disabled for this fixture/scratch screening cycle.
- evaluation: both arms hit the 70 s cap; candidate eval NLL was 7.29866, while all smoke `n=24` documents timed out. No quality comparison or promotion claim is available.

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, fit_decode_timeout_to_n_times_p50
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
