# Continuous autotrain harness repair: cycle-3 hypothesize feedback race (scheduled session sk4t9p)

**Verdict:** reproducible `harness_family=autoresearch` bug, not a model failure.
Both cycle 2 arms trained and evaluated to completion on disk, but the
wrapping CLI process was outer-interrupted before it could write the
predecessor `hypothesizer_feedback` artifact cycle 3's `hypothesize` step
requires — a genuine timing/budget bug in the continuous driver, repaired
before any new model hypothesis is attempted.

## Signal

Cycle 3 (`continuous-loop-20260803-continuous-openui-local-8c0b60dd-c3`)
failed at the `hypothesize` step:

```
CYCLE_ERROR ValueError: latest hypothesis matrix has no terminal feedback;
run a matrix member before forming its successor
```

## Root cause

`scripts/run_autotrain_continuous.py:_arm_execution_deadline` computed the
**outer** supervision deadline for the wrapping `python -m scripts.autoresearch
run ... --execute --experiment-wall-seconds <N>` CLI process as `now + N`
seconds — identical to the `--experiment-wall-seconds` value passed to that
same CLI as the **inner** budget for its own grandchild train/eval subprocess.

Whenever an arm's grandchild subprocess consumed close to its full inner
allotment, the wrapping CLI had **zero** wall-clock room left to write its
terminal `experiment_finished` event, diagnosis, and — critically —
`hypothesizer_feedback` artifact before the outer SIGINT (delivered by
`run_bounded_process`'s own interrupt-then-kill supervision) landed
mid-postprocessing.

Both cycle-2 arms (`control`, `component-plan`) hit this: `events.jsonl`
shows `experiment_started` with no matching `experiment_finished`, and
`artifacts/hypothesizer_feedback/` is empty — even though the underlying
training and evaluation completed on disk. `scoreboard.json`, checkpoints,
and `gates.json` are all complete, and the SDLC Phase A classifier (which
reads those disk artifacts directly, not the CLI's in-process events)
correctly classified the cycle **POSITIVE**
([`continuous-openui-local-sk4t9p-c2-results.md`](continuous-openui-local-sk4t9p-c2-results.md)).
Cycle 1's two arms happened to finish their bookkeeping before the outer
interrupt landed (`hypothesizer_feedback` count 2), which is why the bug did
not surface until cycle 3 needed cycle 2's feedback specifically.

## Fix

Reuse `KILL_GRACE_SECONDS` (10s) as headroom added to
`_arm_execution_deadline`'s outer deadline, on top of `arm_wall_minutes * 60`,
while leaving the `--experiment-wall-seconds` argument passed to the CLI
unchanged — the grandchild's own training/eval budget is unaffected. The
outer deadline stays capped by `cycle_deadline - HARNESS_FINALIZATION_RESERVE_SECONDS`,
so the fix cannot exceed the cycle's own wall cap.

## Regression test

- `tests/test_scripts/test_run_autotrain_continuous.py::test_arm_execution_deadline_outlives_experiment_wall_seconds_budget`
  (new) — asserts the outer deadline exceeds the inner
  `--experiment-wall-seconds` budget by exactly `KILL_GRACE_SECONDS`.
- `test_arm_execution_deadline_preserves_finalization_reserve` (updated) —
  the small-`arm_wall_minutes` branch now expects the added headroom.

## Version stamp

`harness.autoresearch.experiment_campaign` bumped `v162 -> v163` in
`src/slm_training/resources/versions.json` (real behavior change, not a
no-bump doc edit).

## Replay status

Cycle 2's component-plan-vs-control measurement (seed 100002) already has
complete, honest fixture evidence on disk despite the missing
`hypothesizer_feedback` artifact — this was an incomplete infrastructure
record, not a model or measurement failure, so it is not re-run. Per the
autotrain loop law, the next cycle after this repair is the driver's
already-queued rank-1 priority: fresh-seed confirmation of the same
`component-plan` hypothesis.

Machine evidence:
[`continuous-openui-local-sk4t9p-c3-hypothesize-feedback-race-harness-repair.json`](continuous-openui-local-sk4t9p-c3-hypothesize-feedback-race-harness-repair.json).
