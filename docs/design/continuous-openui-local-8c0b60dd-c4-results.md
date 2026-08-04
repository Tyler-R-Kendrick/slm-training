# Continuous autotrain: 2026-08-04 (session 8c0b60dd) cycle 4 — decode timeout, measurement incomplete

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `2b6365e6`

**Verdict:** measurement incomplete, not a model result. All 3 smoke records
on both the matched `control` and size-matched `component-edge` arm hit the
`8.0s` decode timeout (`decode_timeout_count=3`), so `structural_similarity`
and every other quality metric are `None`/unavailable on both arms.

| Arm | Params | compiler_ms_mean | decode_timeout_count | Status |
| --- | ---: | ---: | ---: | --- |
| control | 1,766,990 | 23150.1 | 3/3 | incomplete |
| component-edge | 1,766,990 | 23109.1 | 3/3 | incomplete |

## Diagnosis

Compiler wall time has grown across this session's cycles on the CPU-only
`.venv-smoke` container: `~11.9s` (c3 control) → `~12.7s` (c3 component-plan)
→ `~23.1-23.2s` (c4, both arms). Since **both** the control and candidate
regress together in the same cycle, this reads as host-level CPU decode
throughput (thermal/contention on the container, or accumulating compiler
cache growth across in-process cycles) rather than a `component-edge` code
regression — but that is not yet confirmed.

## SDLC Phase A

**Non-positive** (`measurement_incomplete`, `harness_failure` on both arms).
No stack layer. The `repair_harness` action for this cycle is **not yet
acknowledged**: the next continuous-loop invocation should investigate the
compiler/decode wall growth (e.g. whether a larger
`decode-timeout-seconds` budget is warranted for CPU-only continuous
sessions, or whether a real regression exists) before replaying the frozen
`component-edge` arm.

## Next priorities

1. Investigate compiler/decode wall growth and repair or re-budget before
   replaying `c20260804-continuous-openui-local-8c0b60dd-c4-component-edge`
   (rank 1, confidence 0.95, `repair_harness`).
