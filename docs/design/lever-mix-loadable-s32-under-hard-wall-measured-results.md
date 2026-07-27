# Mix loadable s16 vs s32 under hard wall — NOT SHIP

**Honesty:** fixture/scratch smoke n=3. **Not ship.**

## Hypothesis
s32 on `lever_mix_loadable_v1` lifts meaningful vs mix s16 under hard wall + seed multi-rep.

## Results
| arm | steps_done | stopped | meanful median | vals | parse mean | empty mean | max_lat mean |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| mix s16 | 16 | steps | 0.333 | [0.3333333333333333, 0.3333333333333333, 0.3333333333333333] | 1.000 | 0.00 | 30043 |
| mix s32 | 26 | wall_time_budget | 0.333 | [0.3333333333333333, 0.3333333333333333, 0.3333333333333333] | 1.000 | 0.00 | 30204 |

## Decision
**REJECT** — mix s32 (done=26, wall_time_budget) no quality lift meanful 0.333→0.333

Captured: 2026-07-27T18:53:16.293912+00:00
