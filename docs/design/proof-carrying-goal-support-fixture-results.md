# Proof-carrying goal support fixture results

Claim class: **fixture/diagnostic only**. This is bounded wiring evidence, not a model-quality, ship-readiness, global-unrealizability, or production-performance claim.

- Source commit: `5ee0775f572ba03a91ce89b9482fd71cdb48c20a` (`dirty=false`)
- Suite: `n=4`, seed `0`, local CPU/in-process, no model
- Solver bounds: `{"max_backtracks": 64, "max_depth": 8, "max_nodes": 64, "max_tokens": 4096, "max_verifier_calls": 64}`
- Exact action cap: `2`; goal-query cap per arm: `32`
- Canonical raw bundle: `outputs/autoresearch/proof-carrying-goal-support-fixture`
- Manifest: `4fb4a79fa442d22bdf448ec4536f9d269ffd12301aeaf3c65c72636361f4cc0f`
- Canonical result digest: `07fe9e2b2d32baf4d8d6175e15766d8cd947f75ad9269e644f19f84af4f1b79c`
- Deterministic rerun: `PASS`

## Arm results

| arm | coverage | supported | unsupported | unknown | unobserved | selection regret | inadequate under bounds | false hard prune | verifier calls | expanded nodes | wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| structural_support_reference | 0.888889 | 0.777778 | 0 | 0.111111 | 0.111111 | 0 | 0 | 0 | 14 | 16 | 0.009411 |
| goal_support_production_exact_diagnostic | 0.888889 | 0.222222 | 0.555556 | 0.111111 | 0.111111 | 0.25 | 0.25 | 0 | 26 | 29 | 0.042659 |
| goal_support_evaluation_oracle_diagnostic | 0.888889 | 0.0 | 0.0 | 0.888889 | 0.111111 | 0.0 | 0.0 | 0 | 21 | 24 | 0.023379 |
| goal_support_certified_fixture | 0.888889 | 0.222222 | 0.555556 | 0.111111 | 0.111111 | 0.25 | 0.25 | 0 | 37 | 40 | 0.06427 |

## Interpretation

The production-exact diagnostic arm distinguishes the satisfying and violating alternatives, emits a replayable bounded obstruction for the fully observed inadequate case, preserves incomplete/unknown candidates, and reports the candidate-cap case as coverage uncertainty. The evaluation-oracle arm requires `G11`; its mandatory skip remains UNKNOWN and it is never supplied to certified closure.

Certified fixture pruning removed `3` replay-valid compiler-hard UNSUPPORTED actions, produced `1` bounded empty domain, and recorded `0` false hard prunes. Obstruction-core replay rate was `1.0` with mean core size `1.0`.

No checkpoint was created, so `docs/MODEL_CARD.md` and its README summary were intentionally unchanged. No AgentV model-eval bundle was required because this command performs no model evaluation.

Remaining limitation: these are small exact fixtures over terminal completions. Successor work should measure the default-off diagnostic/certified path on real complete OpenUI completion forests without changing the pinned verifier or using fixture evidence for promotion.
