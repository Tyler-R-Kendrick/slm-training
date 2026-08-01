# Continuous promote effect gate (2026-08-01)

## Problem

Continuous promote plumbing (formal proved, dual-arm runs, LeverProof cert v2
`optimum_feedback → continue`) marked champions **`promoted`** even when
held-out primary delta was **0.0**. Vacuous promote bands
(`[0, 1000]` per-mille SS and parse) made cert continue nearly automatic, so
the queue counted process success as quality learning.

## Fix

`dispose_champion_promote` now requires **all** of:

1. Formal preflight `proved` (timeout still → `promotion_inconclusive`)
2. Dual-arm **promotion primary** win: signed improvement **>**
   `promotion_primary.minimum_effect` (default `held_out.structural_similarity`
   / `0.01`), with optional parse non-regression
3. LeverProof cert v2 + locked expectations digest + `optimum_feedback == continue`

Cert continue is **necessary but not sufficient**. Null / insufficient primary
→ `promotion_failed` with
`promote_primary_null_or_insufficient:...` (model/effect reject). Missing dual
metrics when required → `promote_primary_metrics_missing:...`.

Policy knobs (`policy.v1.json` → `promotion_dispose`):

```json
{
  "require_primary_win": true,
  "require_cert_continue": true,
  "require_dual_arm_metrics": true,
  "require_parse_non_regression": true,
  "ignore_ship_insufficient_n_for_climb": true
}
```

Shared numeric check: `climb_policy.promotion_primary_effect_met` (same
direction / min_effect semantics as Phase A `classify_positive_metrics`).

## Non-goals

- Ship gates / n≥20 / rico on continuous thrash
- Tightening LeverProof bands (follow-up: `metric_expectations.promote.v2`)
- Multi-seed execute under promote wall (document single-seed + min_effect until then)

## Scoreboard honesty

Historical cert-only `promoted` rows are reclassified by
`python -m scripts.repair_vacuous_promotes --root outputs/autoresearch
--loop-id <loop>` → `promotion_failed` with
`historical_reclassification:promoted→promotion_failed:vacuous_cert_null_primary`.
Rows that already show a held-out / promote primary win are kept.
