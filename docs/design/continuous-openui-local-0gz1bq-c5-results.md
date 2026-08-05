# Continuous autotrain: 2026-08-05 (scheduled loop `0gz1bq`) cycle 5 — component-plan decode timeout, measurement incomplete

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c5`
**Integration commit:** `f60785db` (`origin/main` tip `bdf143cd` + cycles 1-4 docs commits)

**Verdict:** infrastructure — measurement incomplete. `control` completed
cleanly; `component-plan` timed out on all 3/3 eval documents at the 36s
decode-timeout ceiling, so no quality comparison exists for this cycle.

## Results

| Arm | status | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) |
| --- | --- | --- | --- | --- | --- |
| control | complete (gate reject) | 0.0 | 0.04307 | 0.82222 | 15070.99 |
| component-plan | **incomplete** (3/3 decode timeout) | — | — | — | — |

Control's `compiler_ms_mean` was ~12.4s, well under the 36s ceiling.
component-plan's incomplete attempts sit around ~34.4s each — consistently
near the ceiling rather than scattered, which reads as a genuine capacity
signal specific to this candidate's heavier decode path on CPU, not random
system-load noise.

## SDLC Phase A

**Non-positive / infrastructure** (`measurement_incomplete`,
`primary_metric_unavailable`). No stacked PR layer; docs + local commit
only. No code/harness change made — the driver's own diagnosis targets
`infrastructure` with `harness_family=null`, i.e. a capacity signal, not a
typed harness defect.

## Hypothesis-history update

Across six independent measurements to date, `component-plan` no longer
reads as a reliable lever at this fixture scale:

| Session/cycle | Result |
| --- | --- |
| `j48f8u` c2 | win (+0.05613) |
| `peuum8` c3 (fresh seed) | tie (0.0) |
| `0gz1bq` c3 (fresh seed) | regression (-0.05833) |
| `0gz1bq` c5 (this cycle) | timeout, no comparison |

## Next priorities

1. (rank 1, confidence 0.95) Replay the identical frozen `control` /
   `component-plan` arms before testing a new hypothesis
   (`c20260805-continuous-openui-local-8c0b60dd-c5-component-plan`).
