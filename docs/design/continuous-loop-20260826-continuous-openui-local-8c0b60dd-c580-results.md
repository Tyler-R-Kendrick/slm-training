# Continuous cycle `continuous-loop-20260826-continuous-openui-local-8c0b60dd-c580`

- loop_id: `continuous-openui-local`
- cycle_index: `580`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260826-continuous-openui-local-8c0b60dd-c580-control:missing_scoreboard, fixture_insufficient_n:c20260826-continuous-openui-local-8c0b60dd-c580-control-latprobe, fixture_insufficient_n:c20260826-continuous-openui-local-8c0b60dd-c580-current-rung-data-heal-latprobe, primary_metric_unavailable, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': 9417.89, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.11787500000000001, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': None, 'smoke.latency_ms_p50': 9417.89, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.11787500000000001, 'smoke.binder_reference_f1': 0.5313492063492063}`

## Replay evidence

- frozen replay source: cycle 579 manifest `d58577bcd9dc9ce34586e88a8e12c9ed8fed4151515ff075691c8a2a226b45cf`
- recipe: evaluation-only reuse, TwoTower candidate with 1,601,794 parameters, CPU/scratch context, seed 100579, smoke `n=24`, 70 s per-arm cap
- candidate: `exit=0`; AgentV smoke assertions 4/9; no new checkpoint was created
- control: `exit=124` before scoreboard; replay measurement therefore remains incomplete
- held-out, adversarial, OOD, and `rico_held` suites were not run; fixture evidence cannot authorize promotion

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
