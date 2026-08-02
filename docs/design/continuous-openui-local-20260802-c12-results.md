# Continuous autotrain cycle 12 results (2026-08-02, loop `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c12` |
| Cycle intent | Investigate/fix the per-arm wall budget, then let the driver consume its pending `retry_measurement` (final attempt, 2/2) |
| Upstream / integration | `b8188a49` / `14ec931b` |
| Device | CPU |

## Investigation: is the ~70s per-arm wall budget a bug or a compute bound?

`_arm_wall_minutes(policy_minutes)` in `scripts/run_autotrain_continuous.py`:

```
arm_seconds = (MAX_HARNESS_WALL_SECONDS(155s) - HARNESS_FINALIZATION_RESERVE_SECONDS(15s)) / 2
            = 70s
```

`MAX_HARNESS_WALL_SECONDS` is itself derived from the repository-wide
`MAX_RUN_MINUTES = 3` invariant (`levers.py`), minus `KILL_GRACE_SECONDS` and
the finalization reserve. This is already the tightest symmetric split of an
immovable ceiling -- there is no slack to reallocate.

`decode_progress.json` telemetry from three independent frozen-replay
attempts (c10, c11, this cycle) converges tightly:

| Cycle | Arm | `processed_record_n` (of 12) | `total_ms_mean` | `compiler_ms_mean` | compiler share |
| --- | --- | --- | --- | --- | --- |
| c10 | both | 2 | -- | -- | -- |
| c11 | control | 3 | 14258.2ms | 13820.1ms | 96.9% |
| c11 | component-structure | 3 | 14233.2ms | 13953.7ms | 98.0% |
| c12 | control | 3 | 14302.7ms | 13892.0ms | 97.1% |
| c12 | component-structure | 3 | 14334.4ms | 14068.3ms | 98.1% |

**Conclusion: decode compute, not budget arithmetic, is the bottleneck.**
~14.2-14.3s/record, >97% of it in `compiler_ms` (the grammar-constrained
forward-symbol-table decode-compiler stage), is stable across three
independent samples. 12 smoke records need ~172s of pure decode -- ~2.5x the
*entire* two-arm wall budget, before training or I/O. No redistribution of
the fixed arm budget closes a deficit this large. At the measured rate, the
theoretical ceiling is ~4 records/arm (untested; would be only a marginal
33% gain over the already-exhausted `n=3` fixture and wasn't judged worth a
new quality-gated data publish for that little signal).

## Harness fix: stop *new* cycles from repeating this mistake

Nothing previously prevented a **new** (non-frozen) screening/promotion
cycle from silently defaulting onto an eval snapshot whose suites cannot
plausibly complete a decode pass in one arm. Added to
`scripts/run_autotrain_continuous.py` (commit `14ec931`,
`harness.autoresearch.experiment_campaign` v60 -> v61):

- `_suite_record_count` -- counts a published suite's records, fails closed
  (`None`) on a missing file.
- `_eval_version_fits_arm_budget` -- estimates total decode time for a
  role's required suites from the measured per-record cost and compares
  against the arm budget minus overhead.
- `_feasible_eval_version` -- prefers `default_eval_version()`'s unfiltered
  choice when it fits; otherwise falls through the same ordered
  `_DEFAULT_EVAL_VERSION_CANDIDATES` (read-only, never mutated) for the
  first candidate that does; never crashes, never returns nothing.

The eval-version call site now uses `_feasible_eval_version(role=role,
arm_wall_minutes=arm_wall_minutes, policy=policy)` instead of the raw
`default_eval_version()`.

**Deliberately scoped to avoid the known pre-existing broken test.** Cycles
10-11 found that editing `engine.py` to durably add `test_data_scaleup_v1`
to `_DEFAULT_EVAL_VERSION_CANDIDATES` drags in
`tests/test_autoresearch/test_climb_policy.py::test_continuous_classify_positive_entry`
via the `check-changed` pre-commit hook's `src/slm_training/autoresearch/`
prefix mapping -- a test that is confirmed already failing on the untouched
baseline and unrelated to this loop. This cycle's fix lives entirely in
`scripts/run_autotrain_continuous.py`, which maps only to
`tests/test_scripts` under `check_changed.py`'s prefix table, so it commits
cleanly:

```
$ python -c "from scripts.check_changed import hook_test_targets, changed_files; \
             print(hook_test_targets(changed_files(staged=True)))"
['tests/test_scripts/test_run_autotrain_continuous.py']
```

Five regression tests added (`test_suite_record_count_reads_lines_and_fails_closed_on_missing`,
`test_eval_version_fits_arm_budget_uses_measured_decode_cost`,
`test_feasible_eval_version_falls_back_when_default_is_oversized`,
`test_feasible_eval_version_keeps_default_when_it_already_fits`,
`test_feasible_eval_version_never_crashes_when_nothing_fits`).
`tests/test_scripts/test_run_autotrain_continuous.py`: **102 passed, 1
skipped**. `verify_version_stamps --check`: clean.

This guard cannot and does not affect the already-frozen c10 replay chain
(frozen replays reuse the exact locked recipe, including `eval_version`, by
design -- see `contracts.md`), so the mandatory `retry_measurement` action
queued in c11's handoff was still consumed this cycle exactly as before.

## The frozen replay: still incomplete, exactly as predicted

The driver was invoked once (`run_autotrain_continuous --supervised
--max-cycles 1 --train-version wf_smoke_v2 --steps 20`). Its own pending
`retry_measurement` action from c11 (final attempt, `1/2`) took priority,
resuming from c10's checkpoints. Both arms hit `wall_timeout` again:
`processed_record_n=3` of 12 for both, matching the root-cause prediction
almost exactly (`compiler_ms_mean` ~14.0-14.1s of ~14.3s total).

`SDLC Phase A`: `positive=false`, `stack_layer=false`,
`action=no_stack_layer_non_positive`. Reasons: `measurement_incomplete`
(both arms, `missing_scoreboard`), `wall_timeout` + `empty_metrics` (both
arms), `primary_metric_unavailable`.

## Driver-detected exhaustion

`max_consecutive_frozen_replays=2` (`policy.v1.json`). This is the second
consecutive incomplete replay of the identical frozen c10 spec (c11 was
`0/2 -> 1/2`, this cycle was `1/2 -> 2/2`, exhausted). The driver correctly
detected exhaustion and emitted:

1. `repair_harness` (owner `improve-openui-harnesses`, `harness_family:
   model_build`, reason: "identical incomplete replay budget exhausted
   (2/2); repair the canonical owner before replaying the frozen arm").
2. `retry_measurement` (owner `autotrain`), explicitly gated on the
   `repair_harness` receipt.
3. `document` (owner `documenting-experiment-results`).

## `repair_harness` disposition: correctly left pending, not forced

The exhaustion diagnosis names `harness_family: model_build` --
`scripts/evaluate_model` and, ultimately, the decode-compiler stage in
`src/slm_training/models/twotower.py` (`compiler_ms`). That is a different
family from this cycle's declared `autoresearch`/`experiments` scope, and it
sits squarely inside the AGENTS.md non-negotiable invariant "constrained
decoding is the product." Safely reducing `compiler_ms` without weakening
the constrained-decode/symbol-table contract needs dedicated profiling and
its own scoped `improve-openui-harnesses` cycle -- not a bolt-on fix here.
This action is **left correctly unacknowledged**. Per
`storage._PREREQUISITE_ACTION_KINDS`, `repair_harness` (like `document`)
blocks cycle 13 from starting via `_require_predecessor_actions` until it
has an evidence-bound receipt. This is the loop's fail-closed exhaustion
contract working as intended, not an oversight.

This cycle acknowledges only the `document` action (this doc pair). The
harness fix that *is* in scope this cycle already landed as a separate,
already-committed `autoresearch/experiments`-family commit (`14ec931`)
unrelated to the specific `repair_harness` action above.

## Recommendation for cycle 13+

1. **infrastructure (blocking, highest leverage):** a future cycle/session
   must invoke `improve-openui-harnesses` against the `model_build` family
   to profile and (if safely possible without weakening the constrained-
   decode invariant) reduce `compiler_ms` decode cost, or explicitly accept
   the `component-structure @ test_data_scaleup_v1` frozen chain stays
   blocked and start a fresh (non-frozen) campaign against a different
   lever/suite instead. Cycle 13 cannot start in this loop until
   `repair_harness` gets a receipt.
2. **infrastructure (shipped this cycle):** `_feasible_eval_version` now
   protects every *future* fresh hypothesis from silently repeating this
   exact class of mistake against an oversized eval snapshot.
3. **model:** `component-structure` remains untested at any suite scale
   with a completed measurement -- still not exhausted, just never
   measured. Any future fresh (non-frozen) screen of it will now
   automatically fall back to a suite that fits (e.g.
   `e938_role_safe_all_targets_v2`, `n=3`) via the new guard, rather than
   attempting `test_data_scaleup_v1` and repeating this cycle's timeout.

## Artifacts

- Harness fix: commit `14ec931` (`scripts/run_autotrain_continuous.py`,
  `tests/test_scripts/test_run_autotrain_continuous.py`,
  `src/slm_training/resources/versions.json`)
- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c12/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c12-{control,component-structure}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-20260802-c12-results.json`
- Predecessor: [cycle 11 results](continuous-openui-local-20260802-c11-results.md)
