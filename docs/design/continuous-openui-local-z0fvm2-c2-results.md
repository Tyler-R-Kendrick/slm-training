# Continuous autotrain: 2026-08-05 (scheduled loop `z0fvm2`) cycle 2 — control arm decode timeout, measurement incomplete

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `abefbb99` (this session's c1 docs commit on top of `bdf143cd`)
**Predecessor:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c1`

**Verdict:** non-positive / inconclusive. The `control` arm hit a typed decode
timeout during evaluation (`decode_timeout_count=3`) and never produced a
primary-metric measurement; the `component-plan` candidate completed but with
null quality (`meaningful_program_rate=0.0`).

## Results

| Arm | exit | train | eval | decode_timeout_count | meaningful_program_rate | latency p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| component-plan | 0 | completed | complete (gate reject) | — | 0.0 | 32136.7 |
| control | 2 | completed | **incomplete** | 3 | — | — |

No primary-metric comparison is possible this cycle (`primary_metric_unavailable`).

## SDLC Phase A

**Non-positive** (`measurement_incomplete`, `harness_failure:experiment_failed`,
`executable_unblock_rejected_low_mpr:mpr=0.0`, `fixture_insufficient_n_alone`).
No stacked PR layer.

No typed `HarnessSignalV1` was emitted for the decode timeout
(`harness_signals=[]` in the campaign JSON) — self-heal per the absolute loop
law queues a `retry_measurement` (frozen replay of the identical arm pair)
rather than routing this to `improve-openui-harnesses` on a single
observation.

## Next priorities

1. (rank 1, confidence 0.95) Replay the exact frozen pair
   (`c20260805-continuous-openui-local-8c0b60dd-c2-component-plan` +
   matched control) once to distinguish a causal termination-supervision
   runtime effect from a one-run timing artifact.

## Honesty

Fixture (`n=3`) screening; ship gates fail as expected. Not a ship claim.
Not model-comparison evidence — the control measurement is incomplete.
