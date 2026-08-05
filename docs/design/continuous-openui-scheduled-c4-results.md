# Continuous autotrain: 2026-08-05 (scheduled loop `continuous-openui-scheduled`) cycle 4 — `generate_batch_size` schema-drift fix, proven executable unblock

**Loop:** `continuous-openui-scheduled`
**Campaign:** `continuous-loop-20260805-continuous-openui-schedu-1e62ecf9-c4`
**Source commit:** `34111e6e`
**Integration commit:** `5e36624a` (this session's `generate_batch_size` schema fix)

**Verdict:** the loop was blocked on `CYCLE_ERROR` (57x `extra_forbidden` pydantic
validation errors) on every attempt before this session; the fix in `5e36624a`
is a **proven executable unblock** — cycle 4's control arm is the first run in
this loop's history to train, evaluate, and reach an honest ship-gate verdict
end to end. The **model** comparison itself is still inconclusive.

## Root cause

`scripts/run_autotrain_continuous.py`'s screening-role hypothesis builder has
pinned `knobs["generate_batch_size"] = 1` for every screening experiment since
the v178-era continuous cycles (comment: "Screening smoke suites are tiny:
baked generate_batch_size groups every document into one decode chunk,
defeating per-record fair-share timeout redistribution"). But
`src/slm_training/autoresearch/schemas.py`'s `ExperimentKnobs`
(`StrictModel`, `extra="forbid"`) never declared the field, and
`DEFAULT_ALLOWED_KNOBS` omitted it too — even though `generate_batch_size` is
a legitimate, already-wired `model_build` knob
(`src/slm_training/harnesses/model_build/config.py:471`, consumed by
`eval_runner.py`, `evaluate_model.py`, `twotower.py`). Every screening-role
`HypothesisMatrix` for the `twotower` track therefore failed strict
validation with `57 validation errors for HypothesisMatrix` /
`hypotheses.N.experiment.knobs.generate_batch_size: Extra inputs are not
permitted`, and the campaign's own `hypotheses` tuple validator then rejected
the resulting empty list (`Tuple should have at least 5 items after
validation, not 0`).

This session's `continuous-openui-scheduled` loop hit that identical
`CYCLE_ERROR` on 2 consecutive in-process self-heal retries
(`blocker_count=2`, one retry short of the loop law's 3-strikes hard-block
threshold) before diagnosis.

## Fix

Commit `5e36624a`:

- Declared `generate_batch_size: int | None` (`ge=1, le=1024`, matching
  `batch_size`'s bounds) on `ExperimentKnobs`.
- Added `"generate_batch_size"` to `DEFAULT_ALLOWED_KNOBS`.
- Added `test_generate_batch_size_knob_is_declared_and_allowlisted` (pins the
  specific field) and
  `test_experiment_knobs_fields_stay_synced_with_default_allowed_knobs` (a
  `model_fields` <-> `DEFAULT_ALLOWED_KNOBS` parity check, modulo the one
  deliberately-excluded legacy read-only field `screening_regime_epoch`) so
  this class of drift can't silently regress again.
- `harness.autoresearch.experiment_campaign` `v180 -> v181`.

`pytest -q tests/test_autoresearch` (269 tests) and the changed-file target
both pass; `scripts.verify_version_stamps --check`,
`scripts.refresh_test_cases --check --changed`, and `scripts.repo_policy`
are all green.

## Proof of unblock (this cycle)

Replaying the identical loop after the fix (`--loop-id
continuous-openui-scheduled --supervised --max-cycles 1`) produced a valid
5+-candidate `HypothesisMatrix` and ran the control experiment to completion
(`exit=0`) for the first time in this loop's history:

| Metric | Control (`c4-control`) |
| --- | --- |
| `trainable_params` | 1,608,960 |
| `smoke.structural_similarity` (primary) | 0.416667 |
| `smoke.meaningful_program_rate` | 0.333333 |
| `smoke.binder_reference_f1` | 0.952381 |
| `smoke.compiler_ms_mean` | 28,335 |
| `smoke.latency_ms_p50` | 29,809.4 |
| Ship gates | **fail** (`expected_gate_rejection=true`) |

Ship-gate failure is expected and honest: fixture `n=3` smoke suite (need
≥20), and `held_out`/`adversarial`/`ood`/`rico_held` suites are absent by
design for a screening-role fixture cycle. This is not a ship claim.

## What is still incomplete

- The candidate `steps` arm **did not run** this cycle
  (`measurement_incomplete: missing_scoreboard`) — the driver queued an
  automatic `retry_measurement` replay of the identical frozen manifest for
  the next supervised invocation.
- `non_regression_fail: binder_reference_f1 0.952381 -> 0.822222` against a
  prior comparator and `primary_metric_null_or_worse` (control `0.416667` vs.
  a `candidate=0.51` reference) mean **no model win is claimed** — this cycle
  proves the harness works again, not that any knob change improves quality.

## SDLC Phase A

- **Model result: non-positive.** Measurement incomplete, no primary metric
  win, no stack layer for a training result.
- **Harness fix: positive (proven executable unblock).** Per
  `autotrain-iteration-delivery`, a proven executable unblock is one of the
  three qualifying triggers for a stacked layer (metric win / ship-quality
  win / proven executable unblock), independent of whether the underlying
  model cycle itself was positive. This session's harness-fix commit
  (`5e36624a`) plus this doc ship as their own PR.

## Next priorities

1. Replay the exact frozen control and `steps` candidate
   (`retry_measurement`, already queued by the driver) before testing a new
   model hypothesis — do not draw a model conclusion from this cycle's
   partial arms.
