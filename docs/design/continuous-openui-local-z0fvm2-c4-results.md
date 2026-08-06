# Continuous autotrain: 2026-08-05 (scheduled loop `z0fvm2`) cycle 4 — component-plan champion falsified on fresh-seed confirmation

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `139b4ad0` (this session's c3 docs commit)
**Predecessor:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3`

**Verdict:** non-positive. The driver automatically ran a **confirmation**
cycle against the c3 champion fingerprint on a fresh seed, and the primary-
metric win did **not** re-hold.

## Results

| Arm | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- |
| control | 0.3333 | 0.46417 | 0.6333 | 9198.84 | fail (gate reject) |
| confirm (component-plan) | 0.3333 | 0.46417 | 0.6333 | 9632.88 | fail (gate reject) |

Exact tie on every certified quality metric. `smoke.structural_similarity`
improvement: **0.0**.

## SDLC Phase A

**Non-positive** (`confirmation_rejected:primary_quality_not_reheld`,
`primary_metric_null_or_worse`, `fixture_insufficient_n_alone`). No new
stacked PR layer — pushed onto the same open positive layer (PR #1445) that
documented the c3 win being confirmation-tested, since the concern is
continuous.

## Champion falsified

`CHAMPION_STATUS entry_id=champ-continuous-openui-local-3-6cfba5d6fd08579f
status=rejected`. The c3 `candidate_queued` champion is now **rejected**,
not promoted. This is consistent with the same `component-plan` candidate's
confirmation history across other sessions of this shared loop (commits
`6d97009`, `528311e`, `7b2f64c` all document the identical rejection
pattern) — screening-scale, single-seed wins on this candidate have
repeatedly failed to survive fresh-seed confirmation.

Note also that both arms here score meaningfully higher than the c1/c3
baselines purely from seed variance (`structural_similarity≈0.46` vs `≈0.33`
previously), which is itself the evidence for why single-seed screening
wins cannot be promotion evidence on their own.

## Next priorities

1. (rank 1, confidence 0.95) The champion fingerprint is exhausted; test a
   distinct size-matched quality hypothesis next (not another `component-plan`
   reproduction).
2. (rank 2, confidence 0.90) Retain training loss as a diagnostic only —
   loss and certified program quality diverged on this confirmation.

## Honesty

Fixture (`n=3`) screening/confirmation evidence only. Not a ship claim.
