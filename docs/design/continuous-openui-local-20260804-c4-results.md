# Continuous autotrain cycle 4 results (2026-08-04)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c4` |
| Cycle intent | `screening` (component-edge decode weighting hypothesis) |
| Source | `eba6db3044076285581b80cfe5294a2ecbcee8a1` |
| Device | CPU (sandbox, no GPU) |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Decode timeout | 8.0s / document |
| Trainable params | 1,766,990 (both arms, size-matched -- largest pair this loop) |

## Run matrix

| Arm | Status |
| --- | --- |
| c4-control | measurement_incomplete: 3/3 documents hit the 8.0s decode timeout |
| c4-component-edge | measurement_incomplete: 3/3 documents hit the 8.0s decode timeout |

`compiler_ms_mean` was ~23,000ms for both arms -- roughly **3x** the per-document decode-timeout
budget. No scoreboard was produced; `primary_metric` is unavailable.

## Root cause

This is a distinct blocker from cycle 1's AgentV-SDK-missing gap: the harness itself ran
end-to-end (AgentV finalized every record disposition), but the constrained decode step could not
finish inside the 8.0s per-document wall on this session's CPU-only sandbox for a 1,766,990-param
model -- the largest size-matched pair this loop has run. Not confirmed as a code defect this
cycle; queued for a dedicated `improve-openui-harnesses` pass rather than an inline fix, since the
right repair (raise `decode-timeout-seconds` for larger arms, shrink the smoke suite, or bound arm
size on CPU-only sandboxes) needs its own evidence, not a same-cycle patch.

Repair action (`repair_harness`, family `model_build`, frozen manifest
`e5843026820010a38e16e67c279a6dd52a7993cdef6694d4e19a82f332b45373`): acknowledged **blocked** —
real external constraint (no GPU in this sandbox at this arm size), not resolved this cycle.

## Next-run priorities

1. **infrastructure:** replay the identical frozen `c4-control`/`c4-component-edge` arms once
   `decode-timeout-seconds` or arm size is repaired for CPU-only sandboxes (confidence 0.95).

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260804-continuous-openui-local-8c0b60dd-c4/`
- Runs: `.../runs/c20260804-continuous-openui-local-8c0b60dd-c4-control/`,
  `.../runs/c20260804-continuous-openui-local-8c0b60dd-c4-component-edge/`
- JSON twin: `continuous-openui-local-20260804-c4-results.json`
