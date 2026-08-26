# Continuous cycle `continuous-loop-20260826-continuous-openui-local-8c0b60dd-c565`

- loop_id: `continuous-openui-local`
- cycle_index: `565`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260826-continuous-openui-local-8c0b60dd-c565-control:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': 19503.99, 'parse_rate': 1.0, 'meaningful_program_rate': 0.375, 'structural_similarity': 0.2987458333333333, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': 7.450146585720748, 'smoke.latency_ms_p50': 19503.99, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.375, 'smoke.structural_similarity': 0.2987458333333333, 'smoke.binder_reference_f1': 0.5313492063492063, 'smoke.eval_nll': 7.450146585720748}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.

## Harness evidence

- The preregistration payload retained the five required evidence-grounded hypotheses and omitted 54 arms that could not execute in this two-arm campaign.
- Both frozen 70-second arms launched. Both hit the arm interrupt (`exit=124`).
- The control created a local scratch checkpoint but timed out before its scoreboard.
- The candidate created a local scratch checkpoint and complete smoke/AgentV evidence (`n=24`), but failed honest ship gates; the missing matched control keeps the campaign incomplete.
- No model-capacity, gate, arm-budget, or constrained-decode invariant changed.
