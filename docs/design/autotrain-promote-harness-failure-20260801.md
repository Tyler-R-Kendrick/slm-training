# Autotrain promote: matrix formal_claims + harness_failure (2026-08-01)

## Gap

Continuous promote rewrote the promote experiment JSON **after**
`hypothesize` to inject `formal_claims`. `autoresearch run` requires **exact**
hypothesis-matrix membership → promote arm `exit=1`:

```text
ValueError: experiment is not an exact member of the latest hypothesis matrix
```

Control still trained; candidate metrics never existed; cert export reported
`promote_cert_incomplete_metrics` and the queue stored **`promotion_failed`**
— a **false model reject**.

## Fix

1. Attach `formal_claims` on the promote experiment **in `_matrix` before
   hypothesize** so the locked matrix already matches execute.
2. Never rewrite experiment files post-lock for formal binding.
3. New champion/ledger outcome **`harness_failure`** for process aborts
   (missing promote run, exit 1, cert incomplete without candidate metrics).
4. Historical reclass via `repair_formal_timeout_history` (pass 2).

## Related

- Formal timeout incomplete → `promotion_inconclusive` (#1251)
- History timeout reclass (#1252)
