# Unfused LTR constrained feedback — 2026-07-15

This candidate used the explicit second-forward teacher-forced LTR objective (`fuse_ltr_loss=false`) on the published `remediated` corpus: 585 records, manifest `928ec8d4921954c7736d2386fe7abf88bbef75523a7cfe404792f45ddcd5d4ba`.

## Result

| Candidate | NLL step 64 | Smoke n | Parse | Structural | Component recall | Placeholder validity | Reward | Timeouts | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `iter-published-remediated-64step-unfused-ltr2-20260715` | 7.2635 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | Reject |

The explicit teacher-forced path completed cleanly but did not improve constrained serialization. Fused LTR, stronger LTR weights, first-token supervision, and unfused LTR have now all failed on the same honest smoke contract. No checkpoint is promoted; the next intervention changes data mixture or target diversity.
