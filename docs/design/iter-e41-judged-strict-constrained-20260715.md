# E41 strict constrained-decoding feedback — 2026-07-15

The judged-corpus checkpoint was evaluated with unconstrained fallback disabled.
This measures the actual output of the constrained parallel path.

| mode | parse | fallback rate | p50 latency |
| --- | ---: | ---: | ---: |
| fallback enabled | 0/3 | 1.000 | 35,212 ms |
| strict constrained | 0/3 | 1.000 attempted | 9,687 ms |

Strict mode confirms that constrained decoding itself does not currently reach a
valid OpenUI program. The fallback was not rescuing quality; it added roughly
25 seconds per example and returned another malformed stream. It is therefore
appropriate to keep fallback available for compatibility but require strict
mode for structural-adherence experiments and ship evaluation.

Evidence: `outputs/runs/iter-e41-judged-20260715/feedback_strict/`.
