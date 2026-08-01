# Continuous autotrain cycle 5 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-c5` |
| Predecessor | `continuous-loop-20260801-c3` (cycle 4 attempt truncated by an undersized outer wall-clock wrapper before either arm finished; no artifacts, not documented as its own cycle) |
| Source | `5bc90d81ff3fc0da24e4d64c1e7b2c8e37f61f1e` |
| Device | CPU |
| Steps | 40 / seed 7 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes per arm; `decode_timeout_seconds=24.0` |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c5-control | `batch_size=2` | 3 | 1.0 | 0.0 | 2646.91 | eval completed; ship gates fail (insufficient n) |
| c20260801-c5-batch1 | `batch_size=1` | 3 | — | — | — | **all 3 smoke decodes hit the 24s timeout**; ship gates fail (insufficient n + no measurable metrics) |

Primary delta: **not comparable** — the candidate arm produced `decode_timeout_count=3`
and every quality/latency metric is `null`.

## Diagnostics

1. `batch_size=1` is a reproducible-looking regression: all 3 smoke-suite
   decodes exhausted the 24s `decode_timeout_seconds` budget, so
   `parse_rate`, `meaningful_program_rate`, `structural_similarity`, and
   `latency_ms_p50` are all `None` in `gates.json`. The control arm
   (`batch_size=2`) decoded normally in the same cycle under the same wall
   cap.
2. `sdlc_delivery.json` recorded `primary_metric_unavailable` for both the
   control/candidate comparison lines — this is the driver correctly
   refusing to compare a real number against `null`, not a classifier bug.
3. An earlier cycle-4 attempt (`continuous-loop-20260801-c4`, not
   documented) was truncated by an outer wall-clock wrapper set too small
   for two sequential 3-minute-capped arms; self-healed by widening the
   outer wrapper before this cycle, per the loop's non-terminating law
   (soft failures/timeouts are inputs to the next cycle, not stop
   conditions).

## Next-run priorities

1. **infrastructure:** re-run `batch_size=1` in isolation to confirm the
   `decode_timeout_count=3` result reproduces before treating it as a typed
   `HarnessSignalV1` (single occurrence so far; repeated-blocker rule needs
   3 identical failures with no new information).
2. **model:** do not conclude anything about `batch_size` and latency from
   this cycle — the candidate arm never produced a measurable decode.
3. **evaluation:** fixture `insufficient_n` (n=3 < 20 on both `smoke` and
   `held_out`) remains expected and non-terminal at `wf_smoke_v2` scale.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c5/`
- Runs: `.../runs/c20260801-c5-control/`, `.../runs/c20260801-c5-batch1/`
- SDLC delivery: `.../sdlc_delivery.json` (`positive=false`,
  `stack_layer=false`, `action=no_stack_layer_non_positive`)
- JSON twin: `continuous-openui-20260730-c5-results.json`
