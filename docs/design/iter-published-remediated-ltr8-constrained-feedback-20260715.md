# LTR weight 8 constrained feedback — 2026-07-15

This candidate trained from the source-controlled `remediated` corpus (585 records, manifest `928ec8d4921954c7736d2386fe7abf88bbef75523a7cfe404792f45ddcd5d4ba`) with compositional output tokens and `ltr_loss_weight=8.0`.

## Recipe

- scratch context, 64 steps, batch size 8, seed 0
- fidelity loss weight 0.5, random masking
- constrained smoke: LTR-primary + repair, 64-token cap, 20-second timeout

## Result

| Candidate | NLL step 64 | Smoke n | Parse | Structural | Component recall | Placeholder validity | Reward | Timeouts | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `iter-published-remediated-64step-ltr8-20260715` | 7.7707 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | Reject |

The constrained decoder completed all cases, so this is a model-quality failure rather than a timeout artifact. Increasing LTR supervision from the control range to 8.0 did not repair first-token/serialization competence. No checkpoint is promoted; the next intervention must change the objective or data signal.
