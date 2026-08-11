# Proof-carrying goal support fixture results

Claim class: **fixture/diagnostic only**. This is bounded wiring evidence, not a model-quality, ship-readiness, global-unrealizability, or production-performance claim.

- Source commit: `fec57dbc846b6652af93de9d59871750da39fc99` (`dirty=false`)
- Suite: `n=4`, seed `0`, local CPU/in-process, no model
- Solver bounds: `{"max_backtracks": 64, "max_depth": 8, "max_nodes": 64, "max_tokens": 4096, "max_verifier_calls": 64}`
- Exact action cap: `2`; goal-query cap per arm: `32`
- Manifest: `08a5cde76499c624700bbd4cd89b9e9b0959ef3e804d37c6bb1dc010ec94db1e`
- Canonical result digest: `10bd4ffca385aafae67f4fbfd853ffafd4bc36067da2548c8aee7fd07cb04f17`
- Deterministic rerun: `PASS`

## Arm results

| arm | coverage | supported | unsupported | unknown | unobserved | selection regret | inadequate under bounds | false hard prune | verifier calls | expanded nodes | wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| structural_support_reference | 0.888889 | 0.777778 | 0 | 0.111111 | 0.111111 | 0 | 0 | 0 | 7 | 8 | 0.020045 |
| goal_support_production_exact_diagnostic | 0.888889 | 0.111111 | 0.666667 | 0.111111 | 0.111111 | 0.25 | 0.25 | 0 | 7 | 8 | 0.047578 |
| goal_support_evaluation_oracle_diagnostic | 0.888889 | 0.0 | 0.333333 | 0.555556 | 0.111111 | 0.0 | 0.0 | 0 | 7 | 8 | 0.040588 |
| goal_support_certified_fixture | 0.888889 | 0.111111 | 0.666667 | 0.111111 | 0.111111 | 0.25 | 0.25 | 0 | 11 | 12 | 0.069188 |

## Interpretation

The production-exact diagnostic arm distinguishes the satisfying and violating alternatives, emits a replayable bounded obstruction for the fully observed inadequate case, preserves incomplete/unknown candidates, and reports the candidate-cap case as coverage uncertainty. The evaluation-oracle arm requires `G11`; its mandatory skip remains UNKNOWN and it is never supplied to certified closure.

Certified fixture pruning removed `3` replay-valid compiler-hard UNSUPPORTED actions, produced `1` bounded empty domain, and recorded `0` false hard prunes. Obstruction-core replay rate was `1.0` with mean core size `1.0`.

No checkpoint was created, so `docs/MODEL_CARD.md` and its README summary were intentionally unchanged. No AgentV model-eval bundle was required because this command performs no model evaluation.

Remaining limitation: these are small exact fixtures over terminal completions. Successor work should measure the default-off diagnostic/certified path on real complete OpenUI completion forests without changing the pinned verifier or using fixture evidence for promotion.
