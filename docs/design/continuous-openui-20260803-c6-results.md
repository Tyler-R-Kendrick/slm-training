# Continuous autotrain: 2026-08-03 cycle 6 — component-inventory exact tie (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c6`
**Integration commit:** `6ca4824` (docs commit from cycle 5, on top of `main`
tip `318492c5`)

**Verdict:** `component-inventory` ties its matched control on every declared
quality metric; non-positive. Fixture screening only.

| Arm | Params | Seed | parse | MPR | structural_similarity | component_type_recall | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,682,360 | 100006 | 1.0 | 0 | .0964 | .08333 | 1164.64 |
| component-inventory | 1,682,360 | 100006 | 1.0 | 0 | .0964 | .08333 | 1119.80 |

Every declared quality metric ties exactly across both arms; p50 latency
differs slightly and does not clear a scored efficiency win.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` suites were not run.

## Screening-bank exhaustion note

This is the fifth screening arm this loop has run against the
`wf_smoke_v2` / `steps=20` recipe (after `bounds` [c1], `component-plan`
fresh-confirm [c3], `batch1` [c4], `component-edge` [c5]) to tie exactly on
the declared primary. The size-matched knob bank at this exact recipe is
increasingly exhausted; the next cycle should prioritize a new preregistered
quality-targeted objective rather than rotating through more knob levers at
the same steps/seed cadence.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). Per `sdlc`
autotrain-iteration-delivery: docs-only local commit, no stacked layer.
Checkpoints are local scratch (`sync_checkpoints=false`), never
reusable/promotable/syncable/shippable; logged in
[`docs/MODEL_CARD.md`](../MODEL_CARD.md) checkpoint history and the README
summary per the model-card duty for created checkpoints.

## Next priorities (ranked by the driver)

1. Prioritize a new preregistered structural or meaningful-quality objective
   before recycling more exhausted size-matched knob arms at this recipe.
2. Keep the matched control as the size-matched baseline every cycle.
3. Soft ship-gate fails on fixture `n` never stop the continuous loop.

Machine evidence:
[`continuous-openui-20260803-c6-results.json`](continuous-openui-20260803-c6-results.json).
