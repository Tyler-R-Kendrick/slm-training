# Harness incomplete ≠ invalid experiment (2026-08-03)

## Law

```text
Harness / infrastructure incompletes are not model results.
They never permanently invalidate an experiment or thrash approach.
Retry the same recipe after the harness is fixed.
```

| Outcome | Model evidence? | Approach status | Retry? |
| --- | --- | --- | --- |
| Complete dual-arm null / quality reject | Yes | closed / `promotion_failed` / `rejected` | Only new seeds per multi-seed policy |
| Formal timeout | No | `promotion_inconclusive` | Yes |
| Missing run, deadline_reserve skip, cert incomplete because arms never ran, process abort | No | **`harness_failure`** | **Yes, after harness fix** |
| Attempt cap while only harness-blocked | No | stay **`harness_failure`** (parked) | Yes after fix — **never** `promotion_failed` |

## Fixes

1. **Refund promote attempts** on harness failure and formal timeout (incomplete ≠ spent).
2. **Attempt cap** on a harness-blocked head parks as `harness_failure` with
   `promote_harness_parked:incomplete_not_model_reject` — does not
   `promotion_failed`.
3. **`_reopen_harness_blocked_champions`**: when `integration_commit` advances past
   `harness_failure_integration_commit`, rearm to `confirmed` for promote retry.
4. **Thrash multi-seed close** ignores deliveries with harness/incomplete reasons
   even if a buggy delivery claimed `measurement_complete`.
5. **Causal-family terminal skip** ignores harness-only `promotion_failed` rows.

## Related

- [`autotrain-promote-deadline-bank-exhaust-20260803.md`](autotrain-promote-deadline-bank-exhaust-20260803.md) — budget margin that *prevents* deadline_reserve skips
- Climb policy promote dispositions table — `harness_failure` is retryable
