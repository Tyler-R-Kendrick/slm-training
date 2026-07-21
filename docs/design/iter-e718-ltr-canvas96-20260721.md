# E718 96-symbol LTR canvas

E718 tests one local decode-only runtime lever on the E714 symbol-only
checkpoint. It reduces `grammar_ltr_max_tokens` from 160 to 96 while preserving
the raw grammar-constrained LTR path, one attempt, eight-second per-record
timeout, honest slot contract, and no unconstrained fallback.

| Arm | Parse | Strict v2 | Fidelity | Structure | Recall | Timeouts | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| E714 160-symbol control | 0.3333 | 0.0 | 1.0 | 0.0880 | 0.4167 | 0 | 14.06s / 14.27s | 0/1 |
| E718 96-symbol treatment | 0.3333 | 0.0 | 1.0 | 0.0963 | 0.25 | 0 | 5.29s / 6.80s | 0/1 |

The shorter canvas preserves the one parseable prediction, eliminates empty
predictions, improves structure slightly, and reduces median latency by 62.4%
and p95 by 52.3%. Component recall regresses by 0.1667; meaningful-program v1,
strict v2, reward, and AgentV remain zero. All predictions pass the symbol-only
output verifier.

Decision: retain 96 symbols as a local smoke/performance Pareto lever. Do not
make it the global default, extend it to other suites, or claim model-quality
progress until semantic recall is recovered on a larger matched evaluation.

Machine-readable evidence:
[iter-e718-ltr-canvas96-20260721.json](iter-e718-ltr-canvas96-20260721.json).
