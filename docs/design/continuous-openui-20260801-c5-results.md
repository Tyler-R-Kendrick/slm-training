# Continuous autotrain cycle 5 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c5` |
| Predecessor | `continuous-loop-20260801-c4` |
| Source | `2a64db996754d2ce25edcc0d82ee448f7843cd85` |
| Device | CPU |
| Steps | 44 / seed 100005 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c5-control | batch_size 2 | 3 | 1.0 | 0.0 | 2956.52 | eval completed; ship gates fail (insufficient n + quality) |
| c20260801-c5-batch1 | batch_size **1** | 3 | 1.0 | 0.0 | 8272.30 | eval completed; ship gates fail (same) |

Primary delta (batch1 − control) p50 latency: **+5315.78 ms** (positive = candidate slower).

## SDLC Phase A classification

Non-positive — clean regression (`primary_metric_null_or_worse`), no quality
change. No stacked PR opened. Local commit only.

## Diagnostics

1. `batch_size=1` nearly tripled decode latency at this recipe with no
   `meaningful_program_rate` movement — drop as a candidate.
2. Across c2-c5, `meaningful_program_rate` on unmodified control arms has
   read `1.0` (c2, seed 100002), `0.333` (c3/c4, seeds 100003/100004), and
   `0.0` (c5, seed 100005) at essentially the same step count (~42-44) —
   confirming this metric is highly seed-sensitive at `n=3` and cannot be
   trusted from a single seed, matching the lesson from the c3/c4 rejected
   champion.
3. `bounds`-only and `canvas`-only isolation (flagged as owed since cycle 4)
   is still not executed — the hypothesizer has picked `both` (c3), `confirm`
   (c4), and `batch1` (c5) instead across three cycles. Budget is control + 1
   candidate per cycle, so isolating both levers individually needs 2 more
   dedicated cycles.

## Next-run priorities

1. **model:** drop `batch_size=1`; no upside observed.
2. **model:** still owed — a cycle proposing `bounds`-only vs control, and a
   separate cycle proposing `canvas`-only vs control, both at `steps~=42-44`.
3. **evaluation:** given seed sensitivity of `meaningful_program_rate` at
   `n=3`, any future screening win should budget a same-seed multi-arm
   comparison (control + candidate at the *same* seed, which the driver
   already does) rather than reading across cycles with different seeds.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c5/` (local, not tracked)
- Runs: `.../runs/c20260801-c5-control/`, `.../runs/c20260801-c5-batch1/`
- JSON twin: `continuous-openui-20260801-c5-results.json`
- SDLC delivery ledger: `outputs/autoresearch/continuous-loop-20260801-c5/sdlc_delivery.json`
- Predecessor: `continuous-openui-20260801-c4-results.md`
