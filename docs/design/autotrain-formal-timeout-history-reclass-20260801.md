# Autotrain formal-timeout history reclassification (2026-08-01)

## Problem

Before [#1251](https://github.com/Tyler-R-Kendrick/slm-training/pull/1251), continuous
promote formal preflight used a **~180s** wall and recorded status **`unknown`**
when Lean hit the wall. Disposition then wrote champion
**`promotion_failed`** and learning-ledger **`promotion_failed`**.

Those outcomes are **false rejections**: wall timeout is incomplete
measurement, not a proof refutation or quality miss.

Evidence on loop `continuous-openui-20260730`: formal preflight artifacts for
promote campaigns c1420 / c1448 / c1492 / c1504 / c1512 all show
`duration_seconds ≈ 179.2` (old wall) with status `unknown`.

## Fix (code)

| Item | Value |
| --- | --- |
| Formal wall | **600s**, caller-owned |
| Formal status on wall | **`timed_out`** |
| Champion disposition | **`promotion_inconclusive`** (retryable) |
| Attempt accounting | refund `promote_attempts` |

## Fix (history)

Tool: `python -m scripts.repair_formal_timeout_history`

Rewrites loop-local:

- `loops/<id>/champion_queue.jsonl` — `promotion_failed` → `promotion_inconclusive`
  when reasons/status/duration match old formal-wall pattern
- `loops/<id>/learning_certificate_ledger.jsonl` — `outcome` likewise
- campaign `formal_preflight_status.json` annotated `timed_out` + historical note

Does **not** rewrite content-addressed `artifacts/formal_preflights/<sha>.json`
(digest immutability). Appends audit rows to
`loops/<id>/historical_reclassification.jsonl`.

### Run (continuous worktree)

```bash
cd /tmp/slm-autotrain-continuous-loop
python -m scripts.repair_formal_timeout_history \
  --root outputs/autoresearch \
  --loop-id continuous-openui-20260730
```

## Honesty

- Fixture ship-gate fails and true quality nulls stay **failed/rejected**.
- Only formal-wall mis-records are reclassified.
- Reclassified heads re-enter promote cadence; train still requires **proved** formal.
