# Continuous autotrain: 2026-08-03 cycle 3 — component-plan fresh-seed confirmation REJECTED

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `ce3a9890` (this container's cycle-2 docs commit, on
top of `main` tip `318492c5`)
**Cycle intent:** `confirm` (fresh-seed confirmation of the `component-plan`
champion queued by cycle 2 / PR #1369)

| Arm | Params | Seed | structural_similarity | binder_reference_f1 | MPR | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,755,764 | 100003 | .23083 | .48889 | .3333 | 3053.94 |
| confirm (component-plan) | 1,755,764 | 100003 | .23083 | .26667 | .3333 | 2889.6 |

**Verdict: REJECTED.** On a fresh seed (100003), `component-plan` no longer
beats its size-matched control on the declared primary
(`smoke.structural_similarity`) — the two arms tie exactly at `.23083`, unlike
the `+.05613` win originally observed at seed 100002
([`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md),
corroborated again in
[`continuous-openui-20260803-c2-container2-results.md`](continuous-openui-20260803-c2-container2-results.md)).
Worse, `binder_reference_f1` **regresses** on the candidate (`.48889` →
`.26667`), a non-regression failure. A small latency/`mpr_per_ms` efficiency
signal (+5.7%) is present but is not the primary and does not offset the
failed quality re-hold. Per the driver's confirmation policy (fail
confirmation closed on quality), this outcome **rejects** the champion.

Ship gates fail as expected (`insufficient_n`, missing
`held_out`/`adversarial`/`ood`/`rico_held` suites) — fixture screening only.

## Relation to prior component-plan evidence

This closes the "fresh-seed confirmation" requirement flagged in
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md)
and repeated in cycle c2's next-priorities. `component-plan` had won at
seed 100002 (twice, in two independent containers) but **fails to reproduce**
at seed 100001 and 100003 was rejected here — the original win does not
generalize across seeds at this fixture scale and **must not be promoted**.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`, `non_regression_fail`,
`confirmation_rejected:primary_quality_not_reheld`). No stack layer; local
commit only, per `sdlc` autotrain-iteration-delivery.

## Next priorities (ranked by the driver)

1. `component-plan` champion fingerprint
   `champ-continuous-openui-local-2-e19bda467f7df6df` is now **exhausted**
   (confirmation rejected) — the next cycle must test a distinct,
   size-matched quality hypothesis, not repeat `component-plan`
   (confidence 0.95).
2. Retain training loss as a diagnostic signal only; it diverged from
   certified program quality on this confirmation (confidence 0.90).
3. Run the next non-exhausted batch-size arm only as a runtime diagnostic
   while a new quality-targeted objective is preregistered (confidence 0.65).
4. Prioritize a new preregistered structural/meaningful-quality objective
   before recycling exhausted quality families (confidence 0.75).

Machine evidence:
[`continuous-openui-20260803-c3-results.json`](continuous-openui-20260803-c3-results.json).
