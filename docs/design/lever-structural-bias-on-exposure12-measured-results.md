# structural_bias re-tune on exposure12 quality champ — NOT SHIP

**Honesty:** fixture/scratch smoke n=3. **Not ship.**

## Hypothesis

Re-tune `structural_bias` on `lever_exposure12_v1` seed47 to cut ~30s latency without dropping meaningful 0.67.

## Results

| sb | last_loss | parse | meaningful | reward | empty | lat_p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.25 | 7.189 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 30007.61 |
| 1.5 | 7.189 | 1.0 | 0.6666666666666666 | 0.8523333333333333 | 0 | 30011.11 |
| 2.0 | 7.189 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 30009.77 |

## Decision

**REJECT** — no structural_bias on exposure12 beats sb=1.5 latency without quality loss; sb=1.25 meanful=0.3333333333333333; sb=2.0 meanful=0.3333333333333333

Captured: 2026-07-27T16:45:27.311454+00:00
