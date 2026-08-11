# KERN-07 — Singleton zero-neural-forward policy + runtime refinement (SLM-530)

## Claim

A **valid complete singleton** legal domain admits a bypass policy whose
**neural-ranker forward cost is 0** under the named
`NeuralForwardOnlyModel` forward-count cost (KERN-06). Validity requires the
forced token to equal the sole complete-domain candidate (`[[token]]`).

This replaces the former definitional `ForwardsOptimum` / `Nat.zero_le`
“minimum” framing. **`Nat.zero_le` does not prove implementation optimality.**

## Outside the theorem

| Work | Owner / status |
| --- | --- |
| Grammar transitions | `DecodeStats.dfa_sync_count` — other cost axes |
| Certificate checks | `DecodeStats.solver_verifier_calls` — other cost axes |
| Memory / kernel launches | Unobserved or other models |
| Wall-clock latency | Requires explicit `ThroughputAssumption` (KERN-06) |

## Lean API (`LeverProofLean.ConstrainedDiffusion`)

| Symbol | Role |
| --- | --- |
| `SingletonProof.valid` | complete ∧ forced = sole candidate |
| `SingletonProof.validBool` | executable / mutation surface |
| `singleton_forced_eq_sole_candidate` | forced ↔ sole candidate |
| `policyNeuralCost` | Nat fold under `NeuralForwardOnlyModel` |
| `singletonBypassNeuralTrace` | empty neural-forward policy trace |
| `singleton_policy_neural_cost_zero` | policy cost = 0 |
| `singleton_admits_zero_neural_policy` | ∃ policy with neural cost 0 |
| `neural_forward_model_ignores_*` | scope: non-forward events cost 0 here |

## Runtime refinement (empirical remainder)

Python (`slm_training.formal.singleton_neural_policy`) projects existing
`DecodeStats.forwards_count` through `NeuralForwardOnlyModel` and checks
certified singleton fixtures observe zero neural forwards. Fixture status is
always `empirical_remainder` / `claim_class=fixture` — never promoted above
what Lean refinement supports.

Frozen fixtures:
`src/slm_training/resources/formal/singleton_neural_policy_fixtures.v1.json`.

Mutation checks reject mismatched forced tokens and incomplete singleton
domains.
