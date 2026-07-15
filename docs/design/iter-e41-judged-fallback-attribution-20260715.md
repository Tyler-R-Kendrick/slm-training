# E41 judged-corpus fallback attribution — 2026-07-15

The evaluator now separates final program validity from constrained-path
success. Final parse remains the ship-facing metric; `constrained_fallback_rate`
records how often grammar-constrained generation had to retry unfiltered.

| metric | value |
| --- | ---: |
| final parse rate | 0.000 (0/3) |
| constrained fallback rate | 1.000 (3/3) |
| AgentEvals passed | 0/5 |
| p50 latency | 35,212 ms |

This confirms parse was not measuring the wrong thing. The constrained parallel
path failed for every example, and the final malformed outputs were produced
after fallback. The new metric prevents those two failures from being conflated
in future training decisions.

Evidence: `outputs/runs/iter-e41-judged-20260715/feedback_stats/e41-judged-stats/`.
