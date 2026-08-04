# Continuous autotrain cycle 3 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-continuous-openui-202607-98199209-c1` |
| Source | `64893fc40e98bd81e20daed36b483e71ee383a56` |
| Device | CPU |
| Steps | 20 (21 declared) / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `smoke` (published suite, size-matched) |
| Wall cap | 3 minutes |
| Objective | Improve the certified OpenUI quality primary on a size-matched fixture arm without lowering `parse_rate` |

## Run matrix

| Arm | Levers | Params | smoke n | parse_rate | meaningful_program_rate | binder_reference_f1 | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | bounds off | 1,608,962 | 3 | 1.0 | 0.0 | 0.6333 | 2463.32 | eval completed; ship gates fail (fixture n + quality) |
| bounds | bounds **on** | 1,608,962 | 3 | 1.0 | 0.0 | 0.6333 | 2437.62 | eval completed; ship gates fail (same) |

Primary metric is `smoke.binder_reference_f1` (this cycle's declared objective
primary, not p50 latency). Delta (bounds − control): **0.0** — byte-for-byte
identical quality metrics across arms. Latency moved -25.70 ms (bounds
faster) but is not the primary and is not, alone, a positive result.

## Diagnostics

1. `grammar_completion_bounds=True` produced no measurable change in
   `binder_reference_f1`, `parse_rate`, `meaningful_program_rate`, or
   `structural_similarity` versus the size-matched control on this 20-step
   fixture — consistent with cycle 2's latency-only finding
   (`continuous-openui-20260730-c2-results.md`), now replicated on the
   quality primary after the eval-path fix landed in #1269.
2. Both arms correctly fail ship gates on fixture `n` and quality; this is
   the expected diagnostic outcome for a 20-step smoke fixture, not a
   promotion signal.
3. SDLC Phase A classification: **non-positive**
   (`primary_metric_null_or_worse:smoke.binder_reference_f1:control=0.6333
   candidate=0.6333 improvement=0.0`). No stack layer opened for this cycle
   per `sdlc` autotrain-iteration-delivery.

## Next-run priorities

1. **model:** rotate the thrash recommendation across the lever bank instead
   of re-testing `bounds`-only — a single lever has now shown a null quality
   delta twice.
2. **evaluation:** keep the matched control every cycle to guard against
   recipe drift.
3. **infrastructure:** soft ship-gate fails on fixture `n` never stop the
   continuous loop (observed, confidence 0.80).

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c1/`
- Runs: `.../runs/c20260801-continuous-openui-202607-98199209-c1-control/`,
  `.../runs/c20260801-continuous-openui-202607-98199209-c1-bounds/`
- JSON twin: `continuous-openui-20260730-c3-results.json`
- Handoff: `cycle_handoff.json` (local, not tracked) — priorities carried
  into this doc verbatim
