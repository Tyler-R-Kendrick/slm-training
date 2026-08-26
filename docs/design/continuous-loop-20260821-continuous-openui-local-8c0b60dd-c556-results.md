# Continuous cycle `continuous-loop-20260821-continuous-openui-local-8c0b60dd-c556`

- loop_id: `continuous-openui-local`
- cycle_index: `556`
- role/intent: `screening` / `retry_measurement`
- primary_metric: `smoke.eval_nll`
- positive: **False**
- stack_layer: **False**
- measurement_complete: `False`
- evidence_class: `fixture`
- reasons: measurement_incomplete:c20260821-continuous-openui-local-8c0b60dd-c556-control:missing_scoreboard, measurement_incomplete:c20260821-continuous-openui-local-8c0b60dd-c556-current-rung-data-heal:smoke:incomplete_document_n=24:decode_timeout_count=24, harness_failure:c20260821-continuous-openui-local-8c0b60dd-c556-control:experiment_failed, empty_metrics:e68c9c0b45b9f6329249daf52458979e1659f99ad4ca1ccc1f5c48b7df0dbe2d, executable_unblock_rejected_low_mpr:mpr=None, primary_metric_unavailable
- control_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': None}`
- candidate_metrics: `{'latency_ms_p50': None, 'parse_rate': None, 'meaningful_program_rate': None, 'structural_similarity': None, 'binder_reference_f1': None, 'eval_nll': 10.538305744332844, 'smoke.eval_nll': 10.538305744332844}`

## Hill-climb this cycle

- went well: —
- went wrong: measurement_incomplete, non_positive
- speculate: replay_frozen_or_fit_decode_to_wall, keep_rotating_size_matched_arms, fit_decode_timeout_to_n_times_p50
- deltas: `{}`

## I10 data-rebuild recovery (2026-08-26)

- recipe: local CPU, strict profile, `programspec`, 8 requested unique roots,
  `cap0_tiny_v2.json`, no publish, no checkpoint
- first replay: **failed closed** before collection because the checked-in plan
  pinned `pack.corpus_generator` / `pack.oracle` at v32 while
  `harness.train_data` was v33
- alias-repair diagnostic: 80 records, mean quality 0.9769, fingerprint
  `6fe6c65a…9eaab`; required artifacts were emitted
- disposition: **incomplete / not acknowledged** — the diagnostic was
  dirty-stamped, admitted 0/8 requested ProgramSpec roots, retained 13 open
  synthesis recommendations, and reported hard `insufficient_unique_roots`
  plus `synthetic_anchor_deficit` findings
- next: commit the plan-alias regression fix, replay the identical immutable
  build clean, inspect feedback again, and only then bind the rebuild receipt

Auto-documented by the continuous driver self-heal closeout. Fixture screening only — not a ship claim.
