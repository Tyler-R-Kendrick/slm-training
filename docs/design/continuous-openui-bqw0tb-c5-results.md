# Continuous autotrain cycle 5 results (2026-08-01, loop `continuous-openui-bqw0tb`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-bqw0tb` |
| Campaign | `continuous-loop-20260801-continuous-openui-bqw0tb-f852ac38-c5` |
| Device | CPU |
| Steps | 21 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |
| Wall cap | 3 minutes |

Per the driver's own rotation priority (after c2/c4 found zero
`binder_reference_f1` delta from `grammar_completion_bounds` /
`compact_active_canvas`), this cycle rotated the lever bank to
`batch_size=1` for the `c5-batch1` arm.

## Run matrix

| Arm | Status |
| --- | --- |
| c5-control | completed; ship gates fail on fixture `insufficient_n` (expected) |
| c5-batch1 | **timed out** — `scripts.evaluate_model` exceeded `MAX_RUN_MINUTES` mid-decode and was interrupted |

## Diagnostics

1. `c5-batch1` hit the repository-wide wall-time cap while inside
   `TwoTowerModel._compiler_ltr_decode_batch` →
   `build_completion_forest` → `terminal_witness`, and was interrupted
   (`KeyboardInterrupt`). Per the iron law ("a timed-out, interrupted, or
   killed run is never evidence"), this arm produces **no** metric and is
   **not** a harness defect finding — it is a soft failure.
2. `c5-control` completed normally with the expected fixture ship-gate
   failure (`insufficient_n`).
3. The driver recorded a `retry_measurement` action against the frozen
   `c5-control` / `c5-batch1` manifests.

## Classification (SDLC Phase A)

**Non-positive.** Wall timeout with no metric win on the treatment arm;
fixture `insufficient_n` alone on control. No stack layer.

## Next-run priorities

1. **infrastructure:** replay the identical frozen `c5-control` /
   `c5-batch1` arms.
2. **model:** if `batch_size=1` continues to time out under the 3-minute
   wall cap, that is a soft/expected consequence of a slower decode path at
   this lever value, not a harness defect — consider a lever value less
   likely to blow the wall.
3. **evaluation:** soft ship-gate/timeout fails on fixture `n` never stop
   the continuous loop.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-bqw0tb-f852ac38-c5/`
- JSON twin: `continuous-openui-bqw0tb-c5-results.json`
