# E42 factorized embeddings on judged corpus — 2026-07-15

E42 was retrained for 64 CPU steps on the independently judged corpus
(b6d135be9806c708486f1f09efd5c993bafdbff99d029c1985488c57c0a11ec1) and
evaluated against the remediated smoke suite (n=3).

| evaluation | parse | structural | fidelity | reward | p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| matrix path | 0/3 | 0.000 | 0.000 | 0.000 | — |
| strict constrained path | 0/3 | 0.000 | 0.000 | 0.000 | 9,583 ms |

Strict constrained telemetry recorded 102 denoiser forwards per example,
constrained_fallback_rate=1.0, and AgentV 0/5 passed with no execution
errors. All three predictions failed with no parseable root element.

Decision: reject factorized embeddings as the next intervention. The evidence
points to the lexer-native training/constrained-path supervision boundary, not
the fallback policy, as the current structural-adherence bottleneck.

Evidence:

- outputs/runs/iter-e42-judged-factorized-20260715/qx_e42_factorized/matrix_result.json
- outputs/runs/iter-e42-judged-factorized-20260715/strict/e42-judged-strict/eval_smoke.json

This is a scratch, smoke-only result and is not a ship claim.
