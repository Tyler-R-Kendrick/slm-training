# Autotrain c1865: c2 decode-timeout harness failure, repaired

**Verdict:** measurement-incomplete infrastructure failure, not a model result; harness repaired same cycle.

Scheduled continuous loop `continuous-openui-scheduled-72bc0d`, campaign
`continuous-loop-20260804-continuous-openui-schedu-6699f447-c2`, cycle 2.
The ranked successor to c1864 (`component-plan`, a distinct size-matched
quality hypothesis at 1,755,764 params) never produced a scoreboard: both the
matched control and the component-plan candidate hit
`decode_timeout_count=3/3` on the smoke suite.

**Root cause:** `evaluate_model.py` grants the decode timeout **per-chunk**,
not per-record. `screening_decode_timeout_seconds=8` gave the 3-record smoke
chunk an effective `8s x 3 = 24.0s` total budget, and every one of the 6
documents across both arms was clipped at exactly `8000.33ms`
(`decode_amortized_ms_per_record_p50`) — zero margin, not host variance. This
is the identical mechanism already diagnosed (and fixed, unmerged) in this
repository's open PR #1403.

**Repair (commit `eecb8304`):** recalibrated `screening_decode_timeout_seconds`
`8 → 10` in `policy.v1.json` (`v4 → v5`), bumped
`harness.autoresearch.experiment_campaign` `v178 → v179`, and added
`test_screening_decode_timeout_has_margin_over_observed_record_cap`, which
pins the new value and confirms it stays under the
`_fit_screening_decode_timeout_seconds` arm-wall ceiling (14s/record for
`smoke_n=3` at this policy's `screening_stage_wall_minutes=3`) — so the raise
does not risk exceeding the symmetric screening arm wall. 223 tests pass.

Classified `SDLC_PHASE_A NON_POSITIVE` (`measurement_incomplete`,
`harness_failure`, `fixture_insufficient_n_alone`, `primary_metric_unavailable`).
No stack layer for this cycle; local commit only. The `document` handoff
action for c2 is acknowledged with this doc; `repair_harness` was already
acknowledged with the fix commit.

The next cycle replays the identical frozen c2 arm under the recalibrated
budget (`retry_measurement`, not a new hypothesis) so the `component-plan`
result is measured honestly before any new lever is tried.

Machine evidence: [`autotrain-cycle-1865-c2-decode-timeout-harness-failure-repaired.json`](autotrain-cycle-1865-c2-decode-timeout-harness-failure-repaired.json).
