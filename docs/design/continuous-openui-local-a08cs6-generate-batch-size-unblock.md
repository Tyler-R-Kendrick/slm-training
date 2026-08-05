# Continuous autotrain: 2026-08-05 (scheduled loop `a08cs6`) — harness unblock: `generate_batch_size` knob

**Loop:** `continuous-openui-local`
**Blocked campaigns:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c1`,
`-c2` (upstream commit `34111e6e`)
**Unblocked replay:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3`
(integration commit `421603fc`)

**Verdict:** positive — **executable unblocking**
([autotrain-iteration-delivery.md](../../.claude/skills/sdlc/references/autotrain-iteration-delivery.md)
criterion 3). Full evidence: [`continuous-openui-local-a08cs6-generate-batch-size-unblock.json`](continuous-openui-local-a08cs6-generate-batch-size-unblock.json).

## What broke

Every default (`role="screening"`) continuous cycle crashed before it could
lock an experiment manifest:

```
python -m scripts.run_autotrain_continuous --loop-id continuous-openui-local \
  --supervised --max-cycles 1 --train-version wf_smoke_v2 --steps 20
```

failed at the `scripts.autoresearch research --offline` stage with a pydantic
`ValidationError`: every generated hypothesis's `knobs.generate_batch_size`
was rejected as `extra_forbidden`, leaving 0/N valid hypotheses against the
`min_hypotheses=5` floor (driver exit code 2). Cycles `c1` and `c2` both hit
this — the driver couldn't even reach the execution stage.

## Root cause

`scripts/run_autotrain_continuous.py`'s `_matrix()` bakes
`generate_batch_size = 1` into every `role="screening"` hypothesis's knobs
(the screening fair-share decode-timeout fix from PR #1433 — tiny fixture
smoke suites otherwise let `generate_batch_size` group every document into
one decode chunk, defeating per-record timeout redistribution). But
`generate_batch_size` was never added to `ExperimentKnobs`
(`src/slm_training/autoresearch/schemas.py`, a `StrictModel` with
`extra="forbid"`) or to `DEFAULT_ALLOWED_KNOBS` in the same file. `batch_size`
is a registered knob; `generate_batch_size` — a distinct, real decode/eval
knob also consumed by `src/slm_training/harnesses/model_build/config.py` and
`eval_runner.py` — was not.

This was **not** a new regression from this session: reverting the schema
change and re-running the suite showed **5 pre-existing tests already failing
this way on tip of main** (`34111e6e`):
`test_matrix_confirm_path_same_levers_new_seed`,
`test_matrix_steps_confirm_preserves_distinct_source_control_recipe`,
`test_matrix_thrash_rotation_recommends_non_bounds`,
`test_thrash_matrix_dedupes_compose_vs_static_knob_signatures`,
`test_thrash_matrix_strips_stale_feedback_when_no_live_feedback`.

## Fix

Commit `421603fc` (branch `claude/great-dirac-a08cs6`):

- Registered `generate_batch_size: int | None = Field(default=None, ge=1,
  le=1024)` on `ExperimentKnobs`, matching the existing `batch_size` field.
- Added `"generate_batch_size"` to `DEFAULT_ALLOWED_KNOBS`.
- Added regression test
  `test_screening_role_generate_batch_size_is_a_registered_knob` in
  `tests/test_scripts/test_run_autotrain_continuous.py`, pinned to the exact
  default (no `confirm_levers`) production path that crashed.
- Bumped `harness.autoresearch.experiment_campaign` `v180 -> v181` in
  `src/slm_training/resources/versions.json` with a history note.
- All 5 previously-failing tests now pass; full
  `tests/test_scripts/test_run_autotrain_continuous.py` +
  `tests/test_autoresearch/` (505 passed, 1 skipped) and
  `.githooks/check-changed` are green.

## Replay proof

The identical frozen arm was replayed post-fix as cycle `c3`
(`continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3`,
`upstream_commit=34111e6e`, `integration_commit=421603fc`,
`predecessor_campaign_id=...-c2`). Both arms (`-control`, `-both`) completed
with exit code 0 and a real, honestly-gated fixture scoreboard — see
[`continuous-openui-local-a08cs6-c3-results.md`](continuous-openui-local-a08cs6-c3-results.md)
for the model-level (non-positive) outcome of that cycle.

## SDLC Phase A

**Positive** under the executable-unblocking criterion: a hard,
unrecoverable driver-level blocker (exit code 2, 0 valid hypotheses, no
manifest could be locked) was removed by a canonical-owner code fix, and the
identical arm then completed with a usable scoreboard (replay-proven). This
is independent of cycle `c3`'s own null primary-metric delta, which the
driver's automatic classifier correctly scored non-positive on its own
terms.

Stacked PR layer opened for this fix + both measured-results docs (this file,
its JSON twin, and `continuous-openui-local-a08cs6-c3-results.{md,json}`).
