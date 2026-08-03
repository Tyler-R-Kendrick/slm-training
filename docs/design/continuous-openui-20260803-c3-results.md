# Continuous autotrain: 2026-08-03 cycle 3 (non-positive, efficiency-only)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `9dcfa7e6` (post-merge of
[#1369](https://github.com/Tyler-R-Kendrick/slm-training/pull/1369))

| Arm | Params | parse | MPR | structural_similarity | binder F1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 1.0 | .3333 | .23083 | .7333 | 3232.98 |
| both | 1,608,962 | 1.0 | .3333 | .23083 | .7333 | 2568.16 |

**Verdict: non-positive.** The `both` arm ties its matched control exactly on
the declared primary and every quality metric; only p50 latency moved
(−20.6%), which clears the driver's `mpr_per_ms` efficiency-win threshold
(gain_fraction `.2589` ≥ `.05`) with quality held, but an efficiency-only
win — with zero primary-metric movement — does not clear the stack-layer
gate on its own. Ship gates fail as expected (`insufficient_n`, n=3).

Per `sdlc` autotrain-iteration-delivery: **no stacked PR** — docs-only local
commit. Checkpoints are scratch, never promotable.

## Next priorities (ranked by the driver)

1. `component-plan` quality hypothesis (already showed a real primary win in
   [cycle 2](continuous-openui-20260803-c2-results.md); worth another
   independent seed).
2. Keep the matched control as the baseline every cycle.
3. `both` is now exhausted at this seed; do not re-select without a new
   hypothesis.

Machine evidence:
[`continuous-openui-20260803-c3-results.json`](continuous-openui-20260803-c3-results.json).
