# Continuous autotrain cycles 5-7 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-scheduled` |
| Campaigns | `continuous-loop-20260801-c5` .. `c7` |
| Source | `7878b6edf404b7a72196cb5cfe64f760c8892ee5` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Wall cap | 3 minutes per run |

Continuation of the c1-c4 batch (`continuous-openui-20260801-c1-c4-results.md`) driven by
the same `run_autotrain_continuous.py` loop, `--max-cycles 3` on top of the merged docs commit.

## Run matrix

| Campaign | Knob | Control p50 | Candidate p50 | mpr | Positive |
| --- | --- | ---: | ---: | ---: | --- |
| c5 | batch_size | 3501.41 ms | 18299.83 ms | 0.0 → 0.0 | **No** (much slower) |
| c6 | grammar_completion_bounds | 2725.64 ms | 2674.34 ms | 0.0 → 0.0 | **No** (latency win rejected: `mpr=0.0 < 0.333`) |
| c7 | compact_active_canvas | 3177.80 ms | 3339.90 ms | 0.0 → 0.0 | **No** (slower) |

## Diagnostics

1. **c5** — the `batch_size` candidate ran far slower (+14.8 s p50) than control; no
   quality signal to offset it. Poor knob choice for a wall-capped latency screen.
2. **c6** — raw p50 *did* improve (51.3 ms faster), but the driver correctly rejected it as
   non-positive: the quality-aware tradeoff gate requires latency wins to hold
   `meaningful_program_rate >= ~1/3`, and both arms were 0.0. A latency delta with an empty
   program is not a win.
3. **c7** — repeats the c1-c4 pattern: candidate slower than control, mpr 0.0 on both arms.
4. All three cycles are additionally `fixture_insufficient_n` on both arms (20-step smoke
   fixture), independent of the mpr/latency classification above.
5. No stack layer opened for c5-c7 — none are positive per `sdlc autotrain-iteration-delivery`.

## Next-run priorities

1. Drop or resize the `batch_size` knob for latency screens at this wall cap (c5's candidate
   is not size/time-matched in a useful way).
2. Re-test `grammar_completion_bounds` only once `meaningful_program_rate` is reliably above
   0 at this step count — raw latency deltas are not meaningful while mpr floors at 0.0.
3. No cycle in c1-c7 has cleared `fixture_insufficient_n`; consider a higher step or seed
   budget within the wall cap before drawing lever conclusions from this loop.

## Artifacts

- Campaigns: `outputs/autoresearch/continuous-loop-20260801-c5/` .. `c7/`
- Delivery ledger: `outputs/autoresearch/sdlc_delivery_ledger.jsonl` (entries c5-c7)
- JSON twin: `continuous-openui-20260801-c5-c7-results.json`
- Predecessor: `continuous-openui-20260801-c1-c4-results.md`
