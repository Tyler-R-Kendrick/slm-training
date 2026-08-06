# Continuous autotrain: 2026-08-05 (session 8c0b60dd, scheduled run) cycle 3 — fresh-seed confirmation rejects component-plan champion (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `093f9c72` (cycle 2's docs commit, `origin/main` tip unchanged)

**Verdict:** the fresh-seed confirmation run rejects the `component-plan`
champion enqueued in cycle 2. Null primary delta, plus a real regression on
`binder_reference_f1`.

| Arm | Seed | structural_similarity | binder_reference_f1 | reward_score | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100003 | .23083 | .48889 | .5493 | 9119.15 |
| confirm | 100003 | .23083 | .26667 | .2830 | 8997.63 |

Primary delta `0.0` (structural_similarity ties exactly). The efficiency
proxy `mpr_per_ms` moves +1.35%, below the 5% minimum-effect gate. Ship gates
fail as expected on fixture `n=3` (need 20).

## Same rejection as every prior fresh-seed confirmation of this fingerprint

`champ-continuous-openui-local-2-6cfba5d6fd08579f` — the same candidate
fingerprint queued and rejected across the repo's whole prior history of this
lever (most recently in
[`continuous-openui-local-peuum8-c1-results.md`](continuous-openui-local-peuum8-c1-results.md),
session peuum8, commit `6d97009`). The screening-seed win from
[cycle 2](continuous-openui-local-8c0b60dd-c2-results.md) does not hold at an
independent seed; `binder_reference_f1` regresses 0.489 → 0.267 on the
confirm arm relative to control at this seed. This candidate should be
treated as exhausted, not requeued again without a new preregistered
hypothesis.

## SDLC Phase A

**Non-positive** (`confirmation_rejected:primary_quality_not_reheld`). No
stack layer opens. Docs land locally.

## Next priorities

1. Exhaust the `component-plan` fingerprint; do not requeue it again without
   a new preregistered hypothesis (rank 1, confidence 0.95).
2. Retain training loss as a diagnostic only, not a promotion proxy — loss
   and certified program quality diverged on this confirmation (rank 2,
   confidence 0.90).
3. Recent registered quality families are exhausted; prioritize a new
   preregistered structural or meaningful-quality objective before
   recycling them (rank 5, confidence 0.75, speculative).

Machine evidence:
[`continuous-openui-local-8c0b60dd-c3-results.json`](continuous-openui-local-8c0b60dd-c3-results.json).
