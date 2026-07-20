# SLM-135 / EFS4-01: Trailed-assumptions ablation fixture (slm135_plan)

Matrix set: `slm135-trailed-assumptions`
Version: `efs4-01-v1`
Status: **plan_only**
Verdict: **not_run**

## Hypothesis

When a locally legal proposal is wrong, a monotone proposal-contingent state permanently removes valid alternatives that depended on the failed assumption; a trailed controller retracts those facts and recovers with zero false certified prunes.

## Falsifier

Either the production architecture never places proposal-derived facts in irreversible state, or on activated cases both policies recover identical support sets while trailing adds only measurable cost.

## Activation gate

Frontier/natural cases run only when an earlier readout establishes an activated branching/recovery regime. This fixture uses only the closed finite benchmark and injected microcases.

## Rows

| Arm | Policy | Seed | Status | Decisions | Backtracks | False prune | Leaked deductions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trail | certified_trail | 0 | plan_only | 0 | 0 | False | 0 |
| certified_only | certified_only_no_branch | 0 | plan_only | 0 | 0 | False | 0 |
| monotone | monotone_proposal | 0 | plan_only | 0 | 0 | False | 0 |
| partial | partial_retract | 0 | plan_only | 0 | 0 | False | 0 |

## Verdict interpretation

* ``trail_required`` — unsafe controls exhibited a false prune or leaked deduction that ``certified_trail`` avoided.
* ``certified_only_already_safe`` — the repository architecture never needs reversible decisions for this fixture.
* ``dependency_tracking_required`` — explicit decision rollback is insufficient, but assumption-dependency retraction closes the gap.
* ``production_trail_bug`` — ``certified_trail`` itself produced a false prune (regression).

## Fixture caveat

This is wiring-only evidence. The fixture is a hand-written two-hole CSP with one assumption-dependent deduction rule. It exercises the formal boundary between certified deductions and reversible decisions, but it is not a frontier-scale natural-recovery campaign and makes no ship-gate claim. Frontier/natural cases require the preregistered activation gate.
