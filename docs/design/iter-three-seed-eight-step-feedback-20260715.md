# Iteration: three-seed eight-step feedback (2026-07-15)

Three matched scratch TwoTower runs used batch size `8`, learning rate `6e-4`,
eight steps, and final-only loss feedback on the same 585-record corpus
(manifest `928ec8d4921954c7736d2386fe7abf88bbef75523a7cfe404792f45ddcd5d4ba`).

| seed | target tokens | weighted held-out NLL | bounded smoke parse | structural similarity | reward |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11,122 | 17.410 | 0.00 | 0.00 | 0.00 |
| 1 | 12,258 | 31.841 | 0.00 | 0.00 | 0.00 |
| 2 | 10,460 | 29.176 | 0.00 | 0.30 | 0.00 |

The three-seed mean NLL is **26.142**, sample standard deviation **7.679**,
and range **17.410–31.841**. Generation remains invalid across all bounded
unconstrained checks: every prediction failed parsing, although seed 2 earned
0.30 structural similarity from partial evidence. AgentV and telemetry artifacts
were persisted for every new run. The constrained decoder remains too slow to
score.

Decision: the variance and uniformly failed generation gate do not justify data
reweighting, deletion, or a recipe promotion. Keep the corpus and recipe fixed
while improving constrained generation observability/termination, then rerun a
matched seed-controlled comparison with a meaningful generated scoreboard.
These are scratch diagnostics, not ship claims.
