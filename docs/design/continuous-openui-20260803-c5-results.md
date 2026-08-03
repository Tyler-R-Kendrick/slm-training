# Continuous autotrain: 2026-08-03 cycle 5 — component-edge exact tie (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c5`
**Integration commit:** `bf85344d` (docs commit from cycle 4, on top of `main`
tip `318492c5`)

**Verdict:** `component-edge` ties its matched control on every declared
quality metric; non-positive. Fixture screening only.

| Arm | Params | Seed | parse | MPR | structural_similarity | binder F1 | placeholder_fidelity | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,766,992 | 100005 | 1.0 | 0 | .04307 | .82222 | .72222 | 4852.35 |
| component-edge | 1,766,992 | 100005 | 1.0 | 0 | .04307 | .82222 | .72222 | 4539.33 |

Every declared quality metric ties exactly across both arms; only p50
latency differs slightly and does not clear a scored efficiency win.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` suites were not run.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). Per `sdlc`
autotrain-iteration-delivery: docs-only local commit, no stacked layer.
Checkpoints are local scratch (`sync_checkpoints=false`), never
reusable/promotable/syncable/shippable; no MODEL_CARD change.

## Next priorities (ranked by the driver)

1. The completed non-positive arm is exhausted; test the distinct
   size-matched `component-inventory` quality hypothesis next
   (confidence 0.90).
2. Keep the matched control as the size-matched baseline every cycle
   (confidence 0.70).
3. Soft ship-gate fails on fixture `n` never stop the continuous loop
   (confidence 0.80).

Machine evidence:
[`continuous-openui-20260803-c5-results.json`](continuous-openui-20260803-c5-results.json).
