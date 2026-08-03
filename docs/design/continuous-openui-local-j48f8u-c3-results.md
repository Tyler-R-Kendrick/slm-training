# Continuous autotrain: 2026-08-03 (session j48f8u) cycle 3 — champion fresh-seed confirmation REJECTED

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `d5b0c318` (this session's merged c1-c2 PR #1376, on `main`)
**Cycle intent:** `confirm` (automatic fresh-seed confirmation of a queued
champion candidate)

**Verdict:** **Rejected.** The champion fingerprint queued by this session's
own cycle 2 (`champ-continuous-openui-local-2-e19bda467f7df6df`, the
`component-plan` structural-similarity win documented in
[`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md))
does not reproduce at a fresh seed, and the candidate recipe actively
regresses other quality metrics relative to its own control.

| Arm | Seed | meaningful_program_rate | structural_similarity | binder_reference_f1 | placeholder_fidelity | reward_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 100003 | .3333 | .23083 | .48889 | .38889 | .54933 |
| confirm | 100003 | .3333 | .23083 | .26667 | .22222 | .283 |

`structural_similarity` — the primary metric on which c2 showed a
`+.05613` win — ties **exactly** (`0.23083333333333333` on both arms,
improvement `0`). Worse, the `confirm` arm **regresses**
`binder_reference_f1` (`.48889 → .26667`), `placeholder_fidelity`
(`.38889 → .22222`), and `reward_score` (`.54933 → .283`) against its own
control — a non-regression failure, not merely a null result.
`meaningful_program_rate` ties at `.3333` on both arms.

This directly falsifies the c2 screening win as a robust effect: the
structural-similarity delta observed at seed `100002` (`+0.05613`, matching
two other independent sessions' identical measurements at that same seed)
does not reproduce at a fresh seed, and the candidate recipe costs quality
elsewhere. This is consistent with the separate `c4`-lineage champion (a
related but not identical `component-plan` candidate from a different prior
session), which remains blocked pending harness repair rather than
confirmed — both lines of evidence now point the same direction: this
fixture-scale `component-plan` effect is **not** a robust quality
improvement.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` were not run.

## SDLC Phase A

**Non-positive** (`confirmation_rejected:primary_quality_not_reheld` +
`non_regression_fail:binder_reference_f1`). No stacked PR layer for this
cycle — local commit and docs only.

## Next priorities

1. Champion fingerprint `champ-continuous-openui-local-2-e19bda467f7df6df`
   is now rejected/exhausted — do not re-select it without a new
   preregistered hypothesis.
2. Retain training loss as a diagnostic only, not a promotion proxy — loss
   and certified structural quality diverged on this confirmation.
3. Test a distinct size-matched quality-targeted objective next rather than
   recycling the same `component-plan` family.

Machine evidence:
[`continuous-openui-local-j48f8u-c3-results.json`](continuous-openui-local-j48f8u-c3-results.json).
