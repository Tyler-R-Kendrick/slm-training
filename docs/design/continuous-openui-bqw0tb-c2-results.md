# Continuous autotrain cycle 2 results (2026-08-01, loop `continuous-openui-bqw0tb`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-bqw0tb` |
| Campaign | `continuous-loop-20260801-continuous-openui-bqw0tb-f852ac38-c2` |
| Source | `88cafe52cf400da20404fd5fa7e9978519c19e16` |
| Device | CPU |
| Steps | 21 / batch 2 / seed 100001 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |
| Wall cap | 3 minutes |

This cycle replays the frozen `c1-control` / `c1-bounds` arm manifests
(`retry_measurement`, see
[c1 results](continuous-openui-bqw0tb-c1-results.md)) now that the venv has
`torch==2.5.1+cpu` installed.

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | binder_reference_f1 | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c2-control | bounds off | 3 | 1.0 | 0.0 | 0.6333 | 4838.87 | eval completed; ship gates fail |
| c2-bounds | bounds **on** | 3 | 1.0 | 0.0 | 0.6333 | 5872.59 | eval completed; ship gates fail |

Primary metric (`smoke.binder_reference_f1`) delta (bounds − control): **0.0**
(bit-identical). Latency delta: **+1033.72 ms** (bounds slower).

## Diagnostics

1. Both arms trained and evaluated end-to-end this time — the c1
   torch-missing infra gap is confirmed fixed.
2. `grammar_completion_bounds=True` produced no change in
   `binder_reference_f1` (or any other semantic metric) versus the matched
   control on this size-matched 21-step fixture, and cost meaningfully more
   decode latency.
3. `meaningful_program_rate` is 0.0 on both arms; honest ship gates fail as
   expected at fixture `n=3` — this is not a quality claim either way.

## Classification (SDLC Phase A)

**Non-positive.** Primary metric delta is exactly 0.0 (`primary_metric_null_or_worse`);
no ship-quality win; no executable-unblocking beyond the already-documented
c1 infra fix. No new stack layer opened for this cycle per
`autotrain-iteration-delivery.md`.

## Next-run priorities

1. **model:** rotate the thrash recommendation across the lever bank
   (e.g. `compact_active_canvas`) instead of re-testing `bounds`-only.
2. **evaluation:** keep the matched control as the size-matched baseline
   every cycle.
3. **infrastructure:** soft ship-gate fails on fixture `n` never stop the
   continuous loop; proceed to the next cycle.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-bqw0tb-f852ac38-c2/`
- Runs: `.../runs/c20260801-continuous-openui-bqw0tb-f852ac38-c2-{control,bounds}/`
- JSON twin: `continuous-openui-bqw0tb-c2-results.json`
