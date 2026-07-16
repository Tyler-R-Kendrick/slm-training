# Lexer-native constrained feedback — 2026-07-15

This matched candidate trained from the source-controlled `remediated` corpus (585 records, manifest `928ec8d4921954c7736d2386fe7abf88bbef75523a7cfe404792f45ddcd5d4ba`) with the lexer-native output tokenizer.

## Recipe

- scratch context, 64 steps, batch size 8, seed 0
- LTR loss weight 2.0, fidelity loss weight 0.5
- constrained smoke evaluation with LTR-primary + repair, 128-token cap, 20-second batch timeout

## Result

| Candidate | NLL step 64 | Smoke n | Parse | Structural | Timeouts | Reward | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `iter-published-remediated-64step-lexer-ltr2-20260715` | 7.2525 | 3 | 0.0000 | 0.0000 | 3 | 0.0000 | Reject |

The lexer representation did not improve constrained generation. The timeout is diagnostic batch behavior; the bounded compositional LTR run established that the decoder can complete, while this lexer candidate remained non-parseable and exceeded the 20-second batch budget. Keep `remediated` compositional control as the current candidate. Train telemetry, scoreboard, and AgentEvals artifacts are retained locally under the run directories.
