# Exposure12 multi-seed confirmation — NOT SHIP

**Honesty:** fixture/scratch smoke n=3. **Not ship.**

## Hypothesis
Quality lift from `lever_exposure12_v1` (abstraction_ladder admitted via cap=12) is stable across seeds 42/47/51 under champion hparams.

## Results
| seed | last_loss | parse | meaningful | reward | empty | lat_p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 10.193 | 0.6666666666666666 | 0.3333333333333333 | 0.6246666666666667 | 1 | 26873.82 |
| 47 | 7.189 | 1.0 | 0.6666666666666666 | 0.8523333333333333 | 0 | 30011.11 |
| 51 | 13.885 | 1.0 | 0.3333333333333333 | 0.5379999999999999 | 0 | 30117.6 |

### Summary
- success (parse=1 & empty=0): **2/3**
- mean parse=0.889 mean meaningful=0.444 mean reward=0.672

## Decision
**ACCEPT** — exposure12 multi-seed mean meaningful=0.444 success=2/3

Prior wf_smoke_v2 multi-seed under same hparams had mean meaningful ≈0.33.
Captured: 2026-07-27T16:36:34.913384+00:00
