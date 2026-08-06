# Autotrain frozen-replay: `retry_measurement` recovery ignored `area="model_build"` handoffs (harness fix)

**Status:** Fixed. Regression test added. Not a modeling result — infra-only.

## Finding

During the `continuous-openui-local` scheduled session `sched02`
(2026-08-05, branch `claude/great-dirac-h4hi9j`), cycle 2
(`continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2`) hit a
dual-arm decode timeout — the same symptom class as the still-open
[`autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md`](autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md)
Blocker 1, but at seed `100002` on the ordinary `component-plan`/`control`
screening pair (see
[`continuous-openui-local-sched02-c2-results.md`](continuous-openui-local-sched02-c2-results.md)).

The driver correctly diagnosed this and queued a typed `retry_measurement`
action with rank-1 priority `area="model_build"` (a legitimate
`PriorityArea` value — see `src/slm_training/autoresearch/schemas.py:858`)
proposing to replay the identical frozen pair. Per
[`continuous.md`](../../.claude/skills/autotrain/references/continuous.md),
this is exactly the documented, expected self-heal path — not something the
agent should hand-retry.

Consuming that `retry_measurement` on the next 3 consecutive supervised
cycles (c3, c4, c5) instead hard-failed identically every time:

```
ValueError: latest hypothesis matrix has no terminal feedback; run a matrix
member before forming its successor
```
(`scripts/autoresearch.py:637`, raised from `_feedback_context`)

This is **not** a modeling result and **not** self-healed anywhere in
`_self_heal_cycle_error` — it met the repeated-blocker rule (same fingerprint
3x, no new information) and would have required reporting `blocked` had it
not been routed to `improve-openui-harnesses` for a real repair per
`continuous.md` rule 3.

## Root cause

`_feedback_context` (`scripts/autoresearch.py:617`) falls back to
`_recover_incomplete_handoff_feedback` (`scripts/autoresearch.py:644`) when
a lineage campaign's matrix has no terminal feedback but does have a
`cycle_handoff.json`. That recovery function gated on:

```python
infrastructure = tuple(
    priority
    for priority in handoff.priorities
    if priority.area in {"harness", "infrastructure"}
)
if not incomplete or not infrastructure:
    return None
```

but the driver's own diagnosis for a wall/decode-timeout retry legitimately
tags the priority `area="model_build"` (confirmed in
`continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2/cycle_handoff.json`,
priority rank 1). The two-value area allowlist silently excluded that case
(and, by the same logic, every other `PriorityArea` value — `data`,
`researcher`, `model`, `evaluation`, `promotion`, `autoresearch`,
`annotations`, `distill`, `experiments`, `preference`, `quality`, `rl` —
none of which are `"harness"`/`"infrastructure"`), so any
`retry_measurement` whose diagnosis was tagged with one of those areas could
never recover feedback and always raised.

## Fix

`_recover_incomplete_handoff_feedback` now gates recovery on the same typed
signal that already establishes intent — `handoff.reasons` starting with
`measurement_incomplete:`/`harness_failure:` (the existing `incomplete`
check) — plus the presence of a priority carrying a
`proposed_experiment_id`, instead of re-deriving eligibility from a second,
narrower `PriorityArea` allowlist:

```python
retry_priorities = tuple(
    priority
    for priority in handoff.priorities
    if priority.proposed_experiment_id is not None
)
if not incomplete or not retry_priorities:
    return None
```

`diagnosis_target` on the recovered feedback stays hardcoded to
`"infrastructure"` (unchanged) — only the gating condition changed. The
previously-passing `area="infrastructure"` case
(`test_feedback_context_recovers_typed_incomplete_handoff`) is unaffected
since it also sets `proposed_experiment_id`.

## Regression coverage

Added `test_feedback_context_recovers_model_build_incomplete_handoff` to
`tests/test_autoresearch/test_harness.py`, mirroring the existing
`area="infrastructure"` case with `area="model_build"` and a
`measurement_incomplete:` reason — asserts recovery succeeds instead of
raising.

```
pytest -q tests/test_autoresearch/test_harness.py -k feedback_context   # 3 passed
pytest -q tests/test_scripts/test_run_autotrain_continuous.py -k confirm  # 21 passed (unaffected)
pytest -q tests/test_autoresearch tests/test_scripts/test_run_autotrain_continuous.py tests/test_scripts/test_autoresearch_remine.py  # 493 passed, 1 skipped
```

## Version stamps

- `harness.autoresearch.experiment_campaign` v181 → v182 (owns
  `tests/test_autoresearch/test_harness.py`).
- `harness.preference.remine_campaign` v5 → v5, `no-bump:` (owns
  `scripts/autoresearch.py` but this change is unrelated to preference-remine
  behavior).

## Not yet resolved

This fix only unblocks the **recovery/retry mechanics**. It does not
diagnose *why* the dual-arm decode timeout itself occurs (seed-dependent
decode pathology vs. sandbox CPU/wall-budget headroom, per the still-open
Blocker 1 in
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md)).
The next supervised cycle can now actually consume the queued
`retry_measurement` and replay the frozen c2 pair; if that replay itself
times out again, that remains open Blocker-1-class evidence requiring a
dedicated profiling session — not a reason to patch wall-budget or routing
logic speculatively.
