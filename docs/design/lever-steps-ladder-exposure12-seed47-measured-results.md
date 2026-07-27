# SFT steps ladder on exposure12 seed47 — NOT SHIP

**Honesty:** fixture/scratch smoke n=3. **Not ship.**

## Hypothesis

More steps on quality train corpus (`lever_exposure12_v1`) lift meaningful beyond 0.67.

## Results

| steps_req | steps_done | stopped_on | last_loss | parse | meaningful | reward | empty | p50 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 16 | steps | 7.189 | 1.0 | 0.6666666666666666 | 0.8523333333333333 | 0 | 30011.11 |
| 32 | 32 | steps | 4.899 | 1.0 | 0.6666666666666666 | 0.8403333333333333 | 0 | 30009.34 |
| 48 | 43 | wall_time_budget | 6.020 | 1.0 | 0.3333333333333333 | 0.8543333333333333 | 0 | 30018.42 |

## Decision

**REJECT** — more steps on exposure12 do not lift meaningful above s16 quality champ

Note: s48 hit `wall_time_budget` at 43 steps; report actual completed steps.

Captured: 2026-07-27T17:33:06.258810+00:00
