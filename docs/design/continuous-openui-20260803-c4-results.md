# Continuous autotrain: 2026-08-03 cycle 4 (non-positive)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `04f1b1ca` (this container's cycle-3 docs commit, on
top of `main` tip `318492c5`)

| Arm | Params | parse | MPR | structural_similarity | binder F1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 1.0 | .3333 | .41667 | .9524 | 7969.65 |
| batch1 | 1,608,962 | 1.0 | .3333 | .41667 | .9524 | 8250.07 |

**Verdict: non-positive.** The `batch1` knob is size-matched against the
control (`wf_smoke_v2`, `steps=20`, `n=3`, seed 100004) and produces an exact
tie on the declared primary (`smoke.structural_similarity`) and on every other
quality metric; only p50 latency moved (slightly worse on the candidate),
which is not the primary and is not a reliable signal at `n=3`. Ship gates
fail as expected (`insufficient_n`, missing
`held_out`/`adversarial`/`ood`/`rico_held` suites) — fixture screening only,
not a ship claim.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`, `fixture_insufficient_n_alone`).
No stack layer opened; local commit only, per `sdlc` autotrain-iteration-delivery.

## Next priorities (ranked by the driver)

1. `component-edge` quality hypothesis is next in the ranked matrix
   (`c20260803-continuous-openui-local-8c0b60dd-c4-component-edge`).
2. Keep the matched control as the size-matched baseline every cycle.
3. Rotate thrash recommendation across the lever bank (not batch-only).
4. Do not re-select the now-exhausted `batch1` arm without a new hypothesis.

Machine evidence:
[`continuous-openui-20260803-c4-results.json`](continuous-openui-20260803-c4-results.json).
