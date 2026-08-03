# Continuous screening bank: multi-seed close (not regime recycle)

## Problem (real)

The thrash bank hit "exhausted" after **one complete non-positive per arm**.
On the live loop that permanently closed **~47/47** arms; under a
**2-distinct-seed** rule only **~3** arms would close. Fixture n is noisy, so
single-null permanent close is false approach death — not honest anti-thrash.

A later "regime epoch reopen" papered over that by re-running the same closed
slugs with a cosmetic epoch knob. That **hid** bank exhaust instead of fixing
closure policy.

## Fix

1. **Permanent arm close** requires `screening_arm_closure.min_complete_null_seeds`
   (default **2**, aligned with `recipe_null_cap.max_nulls_per_family`).
2. Distinct `seed` values count; same seed re-run does not double-count.
3. A complete **positive** clears the null-seed tally for that slug.
4. **Removed** screening_regime_epoch bank reopen as a fail-forward path.
5. True multi-seed full-bank exhaust still raises
   `registered screening arm bank exhausted…` — that means add a **new
   preregistered** quality objective, not recycle rejects.

## What this is not

- Not ship-gate weakening
- Not vacuous promote reopen
- Not silent recycle of multi-seed-closed approaches
