# Continuous autotrain cycle 2 results (2026-08-05, `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2` |
| Source | `8e33c6cf` (this session's own c1 doc commit) |
| Train | `wf_smoke_v2`, 20 steps |
| Eval | `e938_role_safe_all_targets_v2` |

## Context

Cycle 1's ranked priorities pointed at the size-matched `component-plan`
quality hypothesis next. This cycle runs that pairing.

## Run matrix

| Arm | Levers | Status | smoke n | structural_similarity | latency_ms_p50 |
| --- | --- | --- | ---: | ---: | ---: |
| control | component-plan off | **incomplete (3/3 decode timeouts)** | 3 | — | — |
| component-plan | component-plan **on** | complete | 3 | 0.3828 | 30,226.1 |

No primary-metric delta is attributable: the control arm never produced a
scoreboard. The component-plan candidate's `structural_similarity=0.3828`
matches the prior `c2 component-plan screen` precedent (#1384, #1387,
2026-08-03), but without a completed matched control this cycle cannot
confirm or reject the hypothesis on its own.

## Diagnostics

1. Control hit 3/3 typed decode timeouts (`smoke:decode_timeout_count
   actual=3 need=0`) — a runtime/infrastructure failure, not a model
   result. `measurement_integrity_failures` shows every quality metric as
   `None` for this arm.
2. Component-plan's own executable-unblock disposition is rejected on its
   own terms (`mpr=0.0`), independent of the missing control.
3. Driver classification: `climb_state=inconclusive`, next action
   `retry_measurement` — replay the exact frozen pair to determine whether
   the control-only timeout reproduces.

## Next-run priorities

1. Replay the exact frozen c2 control/component-plan pair once before
   drawing any conclusion about component-plan's structural_similarity
   gain.
2. If the control timeout reproduces on replay, file a typed
   `HarnessSignalV1` for the eval/decode-timeout family rather than
   treating this as a model result.
3. Do not promote, sync, or ship either checkpoint; this cycle is
   inconclusive.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2/`
- JSON twin: `continuous-openui-20260805-8c0b60dd-c2-results.json`
