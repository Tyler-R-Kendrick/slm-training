# Continuous cycle `continuous-loop-20260827-continuous-openui-local-8c0b60dd-c591`

- loop_id: `continuous-openui-local`
- cycle_index: `591`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260827-continuous-openui-local-8c0b60dd-c591-control:missing_scoreboard, fixture_insufficient_n:c20260827-continuous-openui-local-8c0b60dd-c591-current-rung-data-heal-latprobe, primary_metric_unavailable, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': 7297.04, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.1363875, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': None, 'smoke.latency_ms_p50': 7297.04, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.1363875, 'smoke.binder_reference_f1': 0.5313492063492063}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive, fixture_insufficient_n_alone, fixture_volume_gate_ship_only
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, keep_ship_gates_do_not_enqueue_confirm
- deltas: `{}`

## Checkpoint and evaluation recipe

- control: TwoTower, 1,661,698 trainable parameters, CPU, 380 steps, seed 100590, 629 records, 40.11 s train, final loss 5.41978
- control checkpoint: `outputs/autoresearch/continuous-loop-20260827-continuous-openui-local-8c0b60dd-c591/runs/c20260827-continuous-openui-local-8c0b60dd-c591-control/checkpoints/last.pt` (6,692,828 bytes; local fixture/scratch); evaluation timed out
- candidate: reused c590 checkpoint; smoke `n=24`, parse 1.0, meaningful 0.208333, structural 0.136387, binder F1 0.531349, p50 latency 7297.04 ms; honest smoke gates failed
- comparison: unavailable because the control produced no scoreboard and neither arm produced eval NLL
- replay disposition: frozen replay budget exhausted (1/1); retire this residual and consume the next distinct hypothesis

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
