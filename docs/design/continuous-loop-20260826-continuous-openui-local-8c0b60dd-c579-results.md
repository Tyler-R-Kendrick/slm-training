# Continuous cycle `continuous-loop-20260826-continuous-openui-local-8c0b60dd-c579`

- loop_id: `continuous-openui-local`
- cycle_index: `579`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260826-continuous-openui-local-8c0b60dd-c579-control:missing_scoreboard, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': 6532.47, 'parse_rate': 1.0, 'meaningful_program_rate': 0.20833333333333334, 'structural_similarity': 0.11787500000000001, 'binder_reference_f1': 0.5313492063492063, 'eval_nll': 8.687258154862187, 'smoke.latency_ms_p50': 6532.47, 'smoke.parse_rate': 1.0, 'smoke.meaningful_program_rate': 0.20833333333333334, 'smoke.structural_similarity': 0.11787500000000001, 'smoke.binder_reference_f1': 0.5313492063492063, 'smoke.eval_nll': 8.687258154862187}`

## Recipe and evidence

- candidate: TwoTower, 1,601,794 trainable parameters, CPU/scratch context, 328 steps, seed 100579, 90 records, 14.88 s train time
- checkpoint: `runs/c20260826-continuous-openui-local-8c0b60dd-c579-current-rung-data-heal/checkpoints/last.pt` (local scratch; no sync or promotion)
- eval: smoke `n=24`, AgentV; 4/9 assertions passed; held-out, adversarial, OOD, and `rico_held` were not run
- control: bounded timeout (`exit=124`) before a scoreboard; candidate `exit=0`

## Synthesis feedback

- strict build `continuous_i10_continuous_openui_local_c578`: 531 candidates, 162 admitted, 386 rejected; parse fitness `0.9397`; warning `high_rejection_rate`
- the filtered training view kept 90 and dropped 72 records
- dominant producer finding: `lexical_typed_map` admitted 0/32 because every candidate failed parse/contract normalization
- duplication-heavy families included `prompt_paraphrase` (`0.8611` duplicate share), `scope_repair_lexical` (`0.8125`), and `programspec_generated` (`0.7188`)
- emitted data-only candidates include prefix/template concentration and synthetic-anchor-deficit repairs; gates and thresholds were unchanged

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms
- deltas: `{}`

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
