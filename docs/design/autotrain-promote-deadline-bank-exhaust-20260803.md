# Promote deadline_reserve + thrash bank exhaust (2026-08-03)

## Symptoms

1. Promotion cycles (c2216, 2220, …, 2240) bound control/promote arms then
   **`arm_skipped` reason=`deadline_reserve`** with zero runs / no scoreboards.
2. Champion stayed `harness_failure` (`missing_promote_run`) while thrash bank
   multi-seed-closed every other arm; the only open thrash slug
   (`scaffold-prefix`) was skipped because the champion still held it.
3. Screening cycles raised `registered screening arm bank exhausted…` and the
   loop **BLOCKED**.

## Root causes

| Layer | Bug |
| --- | --- |
| Arm budget | `_fit_symmetric_arm_budget` filled remaining wall **exactly**; fit→execute
  check saw remaining µs short (`159.788399 < 159.788414`) and skipped **both**
  arms. |
| Champion ledger | `harness_failure` **refunded** `promote_attempts`, so attempt caps never
  dropped a stuck promote head. |
| Bank selection | Empty thrash bank + open `harness_failure` head → hard block instead of
  spending a promote retry. |

## Fixes

1. **Schedule margin** (`_ARM_BUDGET_SCHEDULE_MARGIN_SECONDS=0.25`) in symmetric
   fit + 1 ms epsilon on the execute deadline check.
2. **Refund only formal timeouts** (`promotion_inconclusive`); harness failures
   consume promote attempts.
3. **`BANK_EXHAUST_PROMOTE_FALLBACK`**: if thrash bank empty but a retryable
   promote head exists, promote instead of raising bank-exhausted.
4. **New size-matched thrash successors** (not recycle of multi-seed-closed
   singles): `scaffold-prefix-structure`, `scaffold-prefix-tail`,
   `component-token-prefix`.

## Honesty

Fixture L2 thrash/promote harness repair only. No ship-gate change. Incomplete
promote runs remain non-evidence.
