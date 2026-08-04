# Continuous autotrain: 2026-08-03 cycle 5 (non-positive)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c5`
**Integration commit:** `5222947e` (this container's cycle-4 docs commit, on
top of `main` tip `318492c5`)

| Arm | Params | parse | MPR | structural_similarity | binder F1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,766,987 | 1.0 | 0.0 | .04307 | .8222 | 4415.73 |
| component-edge | 1,766,987 | 1.0 | 0.0 | .04307 | .8222 | 4409.37 |

**Verdict: non-positive.** The `component-edge` knob is size-matched against
the control (`wf_smoke_v2`, `steps=20`, `n=3`, seed 100005) and produces an
exact tie on the declared primary (`smoke.structural_similarity`) and every
other quality metric; p50 latency is statistically indistinguishable between
arms. Ship gates fail as expected (`insufficient_n`, missing
`held_out`/`adversarial`/`ood`/`rico_held` suites) — fixture screening only,
not a ship claim.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`, `fixture_insufficient_n_alone`).
No stack layer opened; local commit only, per `sdlc` autotrain-iteration-delivery.

## Next priorities (ranked by the driver)

1. `component-inventory` quality hypothesis is next in the ranked matrix
   (`c20260803-continuous-openui-local-8c0b60dd-c5-component-inventory`).
2. Keep the matched control as the size-matched baseline every cycle.
3. Rotate thrash recommendation across the lever bank.
4. Do not re-select the now-exhausted `component-edge` arm without a new
   hypothesis.

Machine evidence:
[`continuous-openui-20260803-c5-results.json`](continuous-openui-20260803-c5-results.json).
