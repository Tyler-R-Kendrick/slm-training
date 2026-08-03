# Continuous autotrain: 2026-08-03 cycle 4 — batch1 quality-tied latency delta rejected (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `db14a0e7` (docs commit from cycle 3, on top of `main`
tip `318492c5`)

**Verdict:** `batch1` ties its matched control on every quality metric and
only improves p50 latency; the harness's efficiency-vs-quality-primary rule
correctly rejects this as non-positive. Fixture screening only.

| Arm | Params | parse | MPR | structural_similarity | binder F1 | placeholder_fidelity | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,960 | 1.0 | .33333 | .41667 | .95238 | .91667 | 9007.93 |
| batch1 | 1,608,960 | 1.0 | .33333 | .41667 | .95238 | .91667 | 7514.40 |

Every declared quality metric ties exactly. Only `latency_ms_p50` improves
(~16.6% faster), which the driver also scores as a 19.9% `mpr_per_ms`
efficiency gain (above its 5% minimum). Despite that efficiency signal, the
climb-policy rule that **an efficiency ratio cannot override the role-owned
quality primary** fires: the declared screening primary
(`smoke.structural_similarity`) shows zero movement
(`primary_metric_null_or_worse`), so the cycle is forced non-positive by
design — this is intentional, not a bug: it prevents a pure latency delta on
an otherwise fully-tied arm from being mislabeled as a quality win.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` suites were not run.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse` overrides the efficiency
signal). Per `sdlc` autotrain-iteration-delivery: docs-only local commit, no
stacked layer. Checkpoints are local scratch (`sync_checkpoints=false`),
never reusable/promotable/syncable/shippable; logged in
[`docs/MODEL_CARD.md`](../MODEL_CARD.md) checkpoint history and the README
summary per the model-card duty for created checkpoints.

## Next priorities (ranked by the driver)

1. The completed non-positive arm is exhausted; test the distinct
   size-matched `component-edge` quality hypothesis next (confidence 0.90).
2. Keep the matched control as the size-matched baseline every cycle
   (confidence 0.70).
3. Soft ship-gate fails on fixture `n` never stop the continuous loop
   (confidence 0.80).

Machine evidence:
[`continuous-openui-20260803-c4-results.json`](continuous-openui-20260803-c4-results.json).
