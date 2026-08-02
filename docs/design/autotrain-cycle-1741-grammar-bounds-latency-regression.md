# Autotrain c1741: AgentV toolchain repair + grammar_completion_bounds screen

**Verdict:** the session's checkout was missing the pinned AgentV SDK
(`node_modules/@agentv/core`), which failed both arms' `--ship-gates`
evaluation with `RuntimeError: AgentV SDK is unavailable`
before a scoreboard was published. Fixed with `npm ci` (the session's
inherited `NODE_OPTIONS` was also malformed and blocked `node` from starting
at all; both were resolved by unsetting/repairing the environment, not by
changing repo code). Both frozen checkpoints were then re-evaluated to
produce a complete measurement: `grammar_completion_bounds` ties the matched
control on every quality metric and is 12.29% slower on completed-document
p50 latency. Rejected.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | Reward | p50 (ms) | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 3/3 | 0 | 1.000 | 0.000 | 0.0575 | 0.6333 | 0.000 | 1,278.91 | fail (fixture n=3) |
| grammar_completion_bounds | 1,608,962 | 3/3 | 0 | 1.000 | 0.000 | 0.0575 | 0.6333 | 0.000 | 1,436.09 | fail (fixture n=3) |

Both arms complete the full 3-record smoke suite with zero decode timeouts —
this is a full, scoreable comparison (not a partial/incomplete measurement).
Ship gates fail on both arms for the expected reasons at this fixture scale:
`smoke:insufficient_n` (n=3, needs ≥20) plus the standard fixture-scale
quality thresholds; this is not a production-readiness claim either way.

## Signals and next run

- Hypothesis (`grammar_completion_bounds reduces smoke latency_ms_p50 versus
  the matched control without lowering parse_rate`) is falsified: quality is
  an exact tie on every metric (parse, meaningful-program, structural
  similarity, binder F1, reward) and latency is 12.29% *slower*, not faster.
- No canonical harness repair is indicated by the model comparison itself.
- The AgentV-SDK gap was this session's checkout state (never ran `npm ci`
  after clone), not a defect in `evaluate_model` or the AgentV integration;
  documenting it here so a future session recognizes the same
  `AgentV SDK is unavailable` error as an environment fix (`npm ci`), not a
  harness or model regression, and doesn't reclassify the retry as
  `measurement_incomplete`.
- Both checkpoints are local fixture-scratch artifacts (`outputs/`,
  gitignored, no sync) — neither reusable, promotable, nor ship.
- Next: continue the loop with a fresh hypothesis; `grammar_completion_bounds`
  is closed as non-positive for this arm shape.

Machine-readable evidence is in
[`autotrain-cycle-1741-grammar-bounds-latency-regression.json`](autotrain-cycle-1741-grammar-bounds-latency-regression.json).
