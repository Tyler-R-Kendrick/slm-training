# ltr_max_tokens=64 multi-seed confirm on exposure12 — NOT SHIP

**Honesty:** fixture/scratch smoke n=3. **Not ship.**

## Hypothesis
`--grammar-ltr-max-tokens 64` latency win (seed47) is multi-seed stable without meanful regression.

## Results
| seed | arm | parse | meaningful | empty | max_lat | sum_lat |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | default | 0.6666666666666666 | 0.3333333333333333 | 1 | 30003 | 74087 |
| 42 | ltr64 | 1.0 | 0.3333333333333333 | 0 | 29670 | 78702 |
| 47 | default | 1.0 | 0.6666666666666666 | 0 | 277005 | 337026 |
| 47 | ltr64 | 1.0 | 0.3333333333333333 | 0 | 30017 | 90030 |
| 51 | default | 1.0 | 0.3333333333333333 | 0 | 190464 | 250605 |
| 51 | ltr64 | 1.0 | 0.3333333333333333 | 0 | 30007 | 84332 |

### Aggregate
- mean meaningful: 0.444 → 0.333
- mean parse: 0.889 → 1.000
- mean sum latency: 220572 → 84355 ms
- mean max latency: 165824 → 29898 ms

## Decision
**PARTIAL_ACCEPT** — latency multi-seed better but meanful 0.444→0.333

### Caveat (determinism)
Re-eval of seed47 with the same checkpoint+ltr64 flags did **not** reproduce the
earlier meanful=0.67 / max≈89s result (this confirm got meanful=0.33 / max≈30s).
Decode path still has residual non-determinism on CPU smoke; treat single-run
quality claims as provisional until locked seeds + decode RNG are verified.

## Next lever
Non-fixture programspec/RICO train under quality+ltr64 recipe, or further LTR wall enforcement.

Captured: 2026-07-27T17:16:21.617025+00:00
