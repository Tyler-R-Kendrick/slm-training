# Organic-balanced mixture feedback — 2026-07-15

This experiment used the persisted `remediated` corpus (585 records) with online family-weighted sampling. The reusable mixture manifest is committed beside the corpus as `mixture_organic_balanced.json`.

Weights: RICO organic 0.50, Awwwards organic 0.20, human-curated 0.15, prompt paraphrase 0.10, layout augment 0.05. Repair/template-heavy exposure is reduced without changing records or importing evaluation gold.

## Result

| Candidate | NLL step 64 | Smoke n | Parse | Structural | Component recall | Placeholder validity | Reward | Timeouts | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `iter-published-remediated-organic-balanced-64step-ltr2-20260715` | 7.1042 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | Reject |

The mixture improves held-out NLL versus the unweighted control, but constrained smoke quality remains zero. This is a documented loss-vs-generation divergence; no checkpoint is promoted. Train telemetry, the mixture hash, scoreboard, and AgentEvals bundle are retained.
