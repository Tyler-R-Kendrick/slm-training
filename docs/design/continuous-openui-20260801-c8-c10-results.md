# Continuous autotrain cycles 8-10 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-scheduled` |
| Campaigns | `continuous-loop-20260801-c8` .. `c10` |
| Source | `93d70fb52e4e76e2448ce2ecbc216c5eab6d0ad7` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Wall cap | 3 minutes per run |

Continuation of c1-c7 (`continuous-openui-20260801-c1-c4-results.md`,
`continuous-openui-20260801-c5-c7-results.md`).

## Run matrix

| Campaign | Knob | Control p50 / mpr | Candidate p50 / mpr | Positive |
| --- | --- | --- | --- | --- |
| c8 | bounds+canvas | 2583.83 ms / 0.333 | 2595.34 ms / 0.333 | **No** (primary metric unavailable, tied) |
| c9 | steps | 9463.63 ms / **1.0** | 12729.14 ms / 0.333 | **No** (slower and lower mpr) |
| c10 | batch_size | 9756.03 ms / 0.0 | 2730.17 ms / 0.0 | **No** (7.0 s latency win rejected: `mpr=0.0`) |

## Diagnostics

1. **c9** is the notable data point: the control arm drew `meaningful_program_rate=1.0`,
   the best single mpr observed across this loop's 10 cycles so far. The `steps` candidate
   still lost on both latency and mpr, so this is a clean non-positive result, but the
   control draw itself is worth a repeat at the same seed to check reproducibility.
2. **c10** repeats the c6 shape: a large raw p50 win (7.0 s) correctly rejected by the
   quality-aware tradeoff gate because `mpr=0.0` on both arms. Every latency-only win seen in
   this loop (c6, c10) has hit the same gate — the loop has not yet found an mpr-positive
   lever.
3. **c8** ran the `held_out.structural_similarity` primary metric family (AgentV-backed,
   available since the c1-c4 self-heal) but it stayed `primary_metric_unavailable` this cycle;
   latency/mpr were tied between arms.
4. No stack layer opened for c8-c10 — none are positive per `sdlc autotrain-iteration-delivery`.

## Next-run priorities

1. Repeat c9's control knobs/seed to check whether `mpr=1.0` reproduces or was a
   fixture-sampling outlier.
2. Stop screening latency-only knobs (bounds, canvas, batch_size) in isolation — three
   cycles now (c6, c10, and the c1-c3 batch) show latency deltas with `mpr` floored at 0.0,
   which the tradeoff gate correctly refuses to call positive. Prioritize levers that can move
   `meaningful_program_rate` itself.
3. Continue the loop with an mpr-focused knob (data mix, longer step budget within the wall
   cap, or a lever from `docs/design/autotrain-climb-policy.md`) for the next batch.

## Artifacts

- Campaigns: `outputs/autoresearch/continuous-loop-20260801-c8/` .. `c10/`
- Delivery ledger: `outputs/autoresearch/sdlc_delivery_ledger.jsonl` (entries c8-c10)
- JSON twin: `continuous-openui-20260801-c8-c10-results.json`
- Predecessors: `continuous-openui-20260801-c1-c4-results.md`, `continuous-openui-20260801-c5-c7-results.md`
