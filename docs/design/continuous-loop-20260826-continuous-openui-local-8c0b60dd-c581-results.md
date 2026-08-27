# Continuous cycle `continuous-loop-20260826-continuous-openui-local-8c0b60dd-c581`

- loop_id: `continuous-openui-local`
- cycle_index: `581`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260826-continuous-openui-local-8c0b60dd-c581-control:missing_scoreboard, measurement_incomplete:c20260826-continuous-openui-local-8c0b60dd-c581-current-rung-data-heal:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`

## Recipe and evidence

- control: TwoTower, 1,661,698 trainable parameters, CPU/scratch context, 398 steps, seed 100581, 629 records, 41.29 s train; checkpoint `runs/c20260826-continuous-openui-local-8c0b60dd-c581-control/checkpoints/last.pt`
- candidate: TwoTower, 1,601,794 trainable parameters, CPU/scratch context, 398 steps, seed 100581, 90 records, 16.06 s train; checkpoint `runs/c20260826-continuous-openui-local-8c0b60dd-c581-current-rung-data-heal/checkpoints/last.pt`
- both arms exited `124` at the 70 s cap before scoreboards; no AgentV bundle or metric result was produced
- the checkpoints are local unevaluated scratch artifacts; no upload, promotion, or ship claim

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
