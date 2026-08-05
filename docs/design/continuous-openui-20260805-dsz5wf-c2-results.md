# Continuous autotrain cycle 2 results (2026-08-05, `continuous-openui-local`, session `dsz5wf`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2` |
| Source | `35d85a4b` |
| Train | `wf_smoke_v2`, 20 steps |
| Eval | `e938_role_safe_all_targets_v2` |

## Context

Cycle 1 (this session) reproduced the already-exhausted `bounds` screen.
This cycle exercises the new `--skip-slugs component-plan,component-edge`
override so the driver's rotation lands on a genuinely untested arm
(`component-inventory`) instead of re-running an already-closed one:

```
THRASH_ROTATE cycle=2 recommended=component-inventory skip=['component-edge', 'component-plan']
```

## Run matrix

| Arm | Levers | Status | Notes |
| --- | --- | --- | --- |
| control | component-inventory off | **incomplete** | `missing_scoreboard` |
| component-inventory | component-inventory **on** | **incomplete** | 3/3 `decode_timeout_count` |

Both arms trained to completion (checkpoints written to
`runs/*/checkpoints/last.pt`), but AgentV reported an internal decode
timeout for every record before either arm's `--ship-gates` scoreboard
could be produced. No primary-metric delta is attributable.

## Diagnostics

1. This is a runtime/infrastructure incomplete, not a model result — the
   driver's own SDLC Phase A classification records
   `measurement_incomplete` for both arms and `primary_metric_unavailable`.
2. Typed handoff routes this to `repair_harness`
   (owner `improve-openui-harnesses`, `harness_family=model_build`,
   frozen manifest `28909340c77d61741bd5a668963653e5baed2abf91f506a0e3763221d6899c7b`)
   before any `retry_measurement` replay.
3. Not attempted in this session: the repair investigation itself. Left as
   an open, evidence-bound handoff action for the next iteration rather
   than rushed within the same turn as the `--skip-slugs` harness fix.

## Next-run priorities

1. Route the `model_build` decode-timeout signal through
   `improve-openui-harnesses` before replaying this frozen c2 pair.
2. After repair (or if replay reproduces a completed measurement without
   repair), `retry_measurement` against the frozen manifest before drawing
   any conclusion about `component-inventory`.
3. Do not promote, sync, or ship either checkpoint; this cycle is
   harness-incomplete.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2/`
- JSON twin: `continuous-openui-20260805-dsz5wf-c2-results.json`
