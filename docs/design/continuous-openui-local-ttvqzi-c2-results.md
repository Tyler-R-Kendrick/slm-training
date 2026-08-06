# Continuous autotrain: 2026-08-05 (scheduled session `ttvqzi`) cycle 2 — frozen retry completes cleanly, bounds rejected as exact tie

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `860f1358` (cycle 1's docs commit, merged onto `origin/main` tip `bdf143cd`)
**Predecessor:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c1`
(measurement incomplete — see
[c1 results](continuous-openui-local-ttvqzi-c1-results.md))

**Verdict:** non-positive, but this time a **complete** measurement (not
incomplete). The driver's `retry_measurement` action replayed the identical
frozen `control`/`bounds` arm pair from c1. Control reused c1's training run
(`FROZEN_TRAIN_REUSE`, since its training half already succeeded in c1 —
only evaluation needed to rerun) and both arms finished cleanly this time
(`exit=0`, `FROZEN_REPLAY_ACK`).

## Results

| Arm | exit | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | 0 | 1.0 | 0.0 | 0.0575 | 0.6333 | 5018.91 | fail (gate reject, `n=3`) |
| bounds | 0 | 1.0 | 0.0 | 0.0575 | 0.6333 | 4977.59 | fail (gate reject, `n=3`) |

Primary metric (`smoke.structural_similarity`) improvement: **0.0** — exact
tie, same as the identical `j48f8u-c1` bounds screen from an earlier session
on this same hypothesis identity. Both arms fail every fixture evidence-volume
and quality-threshold gate at `n=3`, as expected for a 20-step smoke
screening cycle.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`, `primary_metric_null_or_worse`
with `improvement=0.0`). No stacked PR layer. This closes out the `bounds`
hypothesis for this loop lineage — it has now been measured cleanly twice
(`j48f8u-c1`, this cycle) with the identical exact-tie result.

## Next priorities

1. (rank 1, confidence 0.90) Test the distinct size-matched `component-plan`
   quality hypothesis next
   (`c20260805-continuous-openui-local-8c0b60dd-c2-component-plan`) — this
   hypothesis has previously shown a real structural-similarity win in
   session `j48f8u` cycle 2, worth an independent reproduction from this
   session.
2. (rank 2) Keep the matched control as the size-matched baseline every cycle.

Machine evidence:
[`continuous-openui-local-ttvqzi-c2-results.json`](continuous-openui-local-ttvqzi-c2-results.json).
