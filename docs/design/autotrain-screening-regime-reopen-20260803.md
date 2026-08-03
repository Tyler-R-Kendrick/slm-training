# Continuous screening bank: regime reopen (2026-08-03)

## Problem

After every registered thrash arm had a complete non-positive measurement in
lineage, `_select_recommended_slug` raised:

```text
registered screening arm bank exhausted; add a distinct preregistered quality
objective instead of recycling a rejected approach
```

Three consecutive copies of that error hard-blocked the continuous driver
(`state=BLOCKED`, `blocker_count=3`) even though the goal (improve fixture
quality under the wall) remained open.

## Fix

Fail-forward with a **screening regime epoch**:

1. Non-positive thrash closures are scoped to `screening_regime_epoch` on arm knobs.
2. When the active epoch has no open bank arm, continuous bumps
   `loops/<loop_id>/screening_regime.json` (`epoch += 1`), reopens thrash
   rotation, and logs `SCREENING_REGIME_TRANSITION`.
3. Reopened matrices carry `screening_regime_epoch` and a seed offset
   (`+ epoch * 10007`) so the approach identity changes — not a silent
   recycle of the same rejected approach (I14).
4. Residual bank-exhaust `RuntimeError`s are recorded as **soft** failures and
   never alone put the loop in `BLOCKED`.

Confirm/promote frozen recipes are unchanged and still outrank thrash.

## Non-goals

- Weakening ship gates or promotion effect gates
- Claiming ship L3 from continuous thrash
- Deleting lineage evidence of prior nulls (audit remains; only epoch-scoped skip)
