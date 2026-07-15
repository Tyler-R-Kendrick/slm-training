# Constrained evaluation override feedback — 2026-07-15

`evaluate_model.py` previously exposed only `--no-grammar-constrained`. Checkpoints trained with `--no-grammar` therefore could not be evaluated with the intended constrained decoder without changing code; prior parse-0 smoke results were unconstrained diagnostics.

The evaluator now accepts `--grammar-constrained` and preserves checkpoint behavior when no override is supplied.

## Corrected feedback

| Run | Grammar | Smoke n | Parse | Structural | Timeouts | Reward | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `iter-published-remediated-64step-ltr2-constrained-fullsmoke-20260715` | explicit on | 3 | 0.0000 | 0.0000 | 3 | 0.0000 | Reject |

The corrected run used the published `remediated` corpus checkpoint with 8 generation steps, chosen-token verification, skipped exact stream probe, and a 10-second per-decode timeout. Constrained decoding timed out on every smoke case, so no quality claim is made. The timeout signal is now actionable for the next harness/model iteration.
