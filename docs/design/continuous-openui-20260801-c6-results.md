# Continuous autotrain cycle 6 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c6` |
| Predecessor | `continuous-loop-20260801-c5` |
| Source | `2a64db996754d2ce25edcc0d82ee448f7843cd85` |
| Device | CPU |
| Steps | 42 / seed 100006 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c6-control | bounds off, canvas off | 3 | 1.0 | 0.333 | 3774.65 | eval completed; ship gates fail (insufficient n only) |
| c20260801-c6-bounds | bounds **on**, canvas off | 3 | 1.0 | 0.333 | 3854.77 | eval completed; ship gates fail (same) |

Primary delta (bounds − control) p50 latency: **+80.12 ms** (positive = candidate slower).

## SDLC Phase A classification

Non-positive — small clean regression, quality held. No stacked PR. Local
commit only.

## Diagnostics

This is the first cycle to isolate `grammar_completion_bounds` alone
(`compact_active_canvas` off) at `steps=42`. Reading it against the earlier
cycles:

| Cycle | Steps | Seed | bounds | canvas | mpr held? | latency delta vs control |
| --- | ---: | ---: | --- | --- | --- | --- |
| c1 | 21 | 100001 | on | off | mpr=0.0 both (quality-blind) | -262.65 ms (rejected: quality-blind) |
| c3 | 42 | 100003 | on | **on** | 0.333 held | -146.62 ms (screening win, later rejected) |
| c4 (confirm) | 42 | 100004 | on | **on** | 0.333 held | +518.14 ms (reversed) |
| c6 | 42 | 100006 | on | off | 0.333 held | **+80.12 ms** (small clean loss) |

`bounds`-only now reads as a wash-to-small-regression once
`meaningful_program_rate` is actually held (c6), consistent with c1's delta
being quality-blind noise rather than a real effect. This narrows the
rejected c3/c4 "bounds+canvas" combined win toward `canvas` or plain seed
variance — not `bounds`.

## Next-run priorities

1. **model:** still owed — isolate `compact_active_canvas`-only at
   `steps=42` (proposed as `c20260801-c6-canvas`, not executed this cycle;
   budget capped at control + 1 candidate) to complete the attribution
   started here.
2. **model:** deprioritize `grammar_completion_bounds` as a latency lever at
   this recipe scale — two independent seeds (c1 quality-blind, c6
   quality-held) both fail to show a real win.
3. **model:** `c20260801-c6-steps` (steps=84) remains an unexplored lever for
   lifting `meaningful_program_rate` further, still unexecuted.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c6/` (local, not tracked)
- Runs: `.../runs/c20260801-c6-control/`, `.../runs/c20260801-c6-bounds/`
- JSON twin: `continuous-openui-20260801-c6-results.json`
- SDLC delivery ledger: `outputs/autoresearch/continuous-loop-20260801-c6/sdlc_delivery.json`
- Predecessor: `continuous-openui-20260801-c5-results.md`
