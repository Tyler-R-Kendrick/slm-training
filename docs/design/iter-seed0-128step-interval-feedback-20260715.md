# Iteration: interval feedback selects the 64-step checkpoint (2026-07-15)

The 128-step seed-0 recipe was rerun with `--test-dir` and
`--loss-eval-every 32`. Complete held-out loss feedback was persisted at steps
32, 64, 96, and 128.

| step | target tokens | weighted NLL |
| ---: | ---: | ---: |
| 32 | 44,239 | 8.883 |
| 64 | 87,055 | 7.057 |
| 96 | 129,513 | 7.771 |
| 128 | 172,553 | 7.312 |

The harness selected `best_weighted_nll.pt` at step 64. Bounded constrained
decoding reproduced structural similarity **0.2333**, placeholder validity
**0.2667**, and component recall **0.25**, while parse and reward remained **0**.
Loss suites were 39.84% of the 56,979 ms run.

This validates interval checkpoint selection and rejects the final 128-step
checkpoint for generation quality. The CLI now fails fast when interval
evaluation is requested without `--test-dir`, preventing silent missing
feedback. Scratch diagnostic only; no checkpoint promotion or ship claim.
