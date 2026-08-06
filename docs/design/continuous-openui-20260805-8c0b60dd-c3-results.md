# Continuous autotrain cycle 3 results (2026-08-05, `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3` |
| Cycle intent | `retry_measurement` (frozen replay of c2) |
| Source | `5a7c14c3` |
| Train | `wf_smoke_v2`, 20 steps |
| Eval | `e938_role_safe_all_targets_v2` |
| Replay of | `frozen_manifest_sha256=383a9dc5a5c1...` (c2) |

## Context

Cycle 2's control arm hit 3/3 typed decode timeouts while the component-plan
candidate completed with `structural_similarity=0.3828`. The driver queued a
`retry_measurement` to replay the exact frozen pair and determine whether the
control timeout was systematic or a one-run artifact.

## Run matrix

| Arm | Status | smoke n | structural_similarity | Notes |
| --- | --- | ---: | ---: | --- |
| control | **incomplete (3/3 decode timeouts, reproduced)** | 3 | — | Identical failure mode to c2 |
| component-plan | complete (reproduced) | 3 | 0.3828 | Identical to c2's measurement |

## Diagnostics

1. Both cycle-2 outcomes reproduce exactly under an identical frozen config:
   the control-only decode timeout reproduces deterministically, and the
   component-plan candidate's `structural_similarity=0.3828` repeats.
2. Because the timeout reproduced rather than being a one-off, the driver
   treats this as a **resolved infrastructure attribution** (confidence
   1.00, disposition "retire arm; test a distinct hypothesis") rather than
   a harness defect requiring a `repair_harness` action — no
   `HarnessSignalV1` was raised.
3. The component-plan candidate is rejected on its own absolute quality
   (`meaningful_program_rate=0`, `binder_reference_f1=0`), not merely left
   inconclusive for lack of a control.

## Next-run priorities

1. Retire the `component-plan` vs matched-control pairing at this recipe —
   it is now conclusively rejected (reproduced control timeout + rejected
   candidate absolute quality), not merely inconclusive.
2. Test the distinct size-matched `component-edge` quality hypothesis next
   per the driver's ranked priorities.
3. Do not promote, sync, or ship either checkpoint.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3/`
- JSON twin: `continuous-openui-20260805-8c0b60dd-c3-results.json`
