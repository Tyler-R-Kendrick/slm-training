# Continuous autotrain: 2026-08-04 (scheduled loop `peuum8`) cycle 3 — component-plan champion rejected on fresh-seed confirmation

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `ad41bbf0` (`origin/main` tip `0c2c6bc7` + merge)

**Verdict:** non-positive. The `component-plan` win queued in
[cycle 2](continuous-openui-local-peuum8-c2-results.md) does **not**
reproduce on a fresh seed.

## Results

| Arm | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- |
| control | 0.3333 | 0.23083 | 0.48889 | 8331.53 | fail (gate reject) |
| confirm (component-plan) | 0.3333 | 0.23083 | 0.26667 | 8161.84 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement on this fresh
seed: **0.0** — exact tie, not the `+0.05613` observed in cycle 2. Worse,
the confirm arm **regresses** `binder_reference_f1` against its matched
control (0.48889 → 0.26667), a non-regression gate fail. The driver's own
diagnosis: "training loss and certified program quality diverged on the
confirmation; retain loss as a diagnostic, not a promotion proxy."

## Champion disposition

`CHAMPION_STATUS entry_id=champ-continuous-openui-local-2-6cfba5d6fd08579f
status=rejected campaign=continuous-loop-20260804-continuous-openui-local-8c0b60dd-c3`.
The champion candidate queued in cycle 2 is now **rejected and exhausted** —
it will not be requeued without a new preregistered hypothesis.

## SDLC Phase A

**Non-positive** (`confirmation_rejected:primary_quality_not_reheld`,
`non_regression_fail:binder_reference_f1`, `fixture_insufficient_n_alone`).
No stacked PR layer opens; docs + local commit only.

## Next priorities

1. (rank 1, confidence 0.95) Exhaust the rejected champion fingerprint;
   test a distinct size-matched hypothesis next rather than re-running
   `component-plan`.
2. (rank 2, confidence 0.90) Retain training loss as a diagnostic signal
   only — it diverged from certified program quality on this confirmation.
3. (rank 3, confidence 0.65, speculative) Run the next non-exhausted
   batch-size arm as a runtime diagnostic while a new quality-targeted
   objective is preregistered (`c20260804-continuous-openui-local-8c0b60dd-c3-batch1`).
