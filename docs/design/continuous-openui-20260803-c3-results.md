# Continuous autotrain: 2026-08-03 cycle 3 — component-plan confirmation rejected (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `318492c5` (current `main` tip)

**Verdict:** the cycle-2 `component-plan` champion (fingerprint
`champ-continuous-openui-local-2-e19bda467f7df6df`) does **not** replicate at
a fresh seed and is now `rejected`. Fixture screening only — not a ship or
promotion claim.

| Arm | Params | Seed | structural_similarity | meaningful_program_rate | binder_reference_f1 | placeholder_fidelity | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,755,764 | 100003 | .23083 | .33333 | .48889 | .38889 | 2802.57 |
| confirm (component-plan) | 1,755,764 | 100003 | .23083 | .33333 | .26667 | .22222 | 2761.73 |

Primary (`smoke.structural_similarity`) ties exactly (improvement `0.0`), and
`meaningful_program_rate` also ties. Worse, the confirm arm **regresses**
against its own matched control on `binder_reference_f1` (`.48889` → `.26667`)
and `placeholder_fidelity` (`.38889` → `.22222`) — a non-regression failure,
not just a null. The driver's efficiency check on `mpr_per_ms` also rejects
the candidate (`gain_fraction=0.0148 < 0.05` minimum effect).

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` suites were not run.

## Relation to prior component-plan evidence

This is the fresh-seed confirmation that
[`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)
(rank-1 priority, confidence 0.95) called for before any promotion or Lean
preflight. The result: **confirmation rejected**
(`confirmation_rejected:primary_quality_not_reheld`). The cycle-2 structural
win is confirmed to be seed-dependent noise, not a stable effect, and must
not be promoted, replayed as a champion, or carried into a Lean preflight
without a new preregistered hypothesis.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`, `non_regression_fail`,
`efficiency_win_rejected_min_effect`, `confirmation_rejected`). Per `sdlc`
autotrain-iteration-delivery: docs-only local commit, no stacked layer.
Checkpoints are local scratch (`sync_checkpoints=false`), never
reusable/promotable/syncable/shippable; logged in
[`docs/MODEL_CARD.md`](../MODEL_CARD.md) checkpoint history and the README
summary per the model-card duty for created checkpoints.

## Next priorities (ranked by the driver)

1. Exhaust the `component-plan` champion fingerprint; test a distinct
   size-matched quality-targeted objective instead of re-spending steps on it
   (confidence 0.95).
2. Treat training loss as a diagnostic only, not a promotion proxy — loss and
   certified quality diverged on this confirmation (confidence 0.90).
3. Prioritize a new preregistered structural or meaningful-quality objective
   before recycling exhausted quality families (confidence 0.75).

Machine evidence:
[`continuous-openui-20260803-c3-results.json`](continuous-openui-20260803-c3-results.json).
