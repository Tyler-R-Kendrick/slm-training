# First-token LTR supervision feedback — 2026-07-15

The fused LTR objective was changed to always mask and score the first post-BOS token, while retaining random suffix coverage. A focused regression test verifies that position is masked. This candidate trained from the published `remediated` corpus: 585 records, manifest `928ec8d4921954c7736d2386fe7abf88bbef75523a7cfe404792f45ddcd5d4ba`.

## Result

| Candidate | NLL step 64 | Smoke n | Parse | Structural | Component recall | Placeholder validity | Reward | Timeouts | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `iter-published-remediated-64step-ltr2-firsttoken-20260715` | 7.3020 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | Reject |

The regression test passed, and the constrained decoder completed all cases, but every prediction remained non-parseable. The objective fix is retained because it closes a real supervision blind spot; it is not a quality improvement at this budget. No checkpoint is promoted.
