# SLM-220: activation-side causal restriction energy

**Verdict:** `rejected`

**Report hash:** `5de0f767ff3844b1074c17c3f0b60e6a38ee45ec0ceea40e0ec84576392a5312`

**Semantic floor:** `7839ef6b6e37710d487757da9170017d7b76a9d12ca1fb314bdb0fa23a4dd83d` (`inconclusive`)

The estimator contract is supported on analytic systems. The requested model-retrospective hypothesis is rejected for use because no compatible checkpoint/state-manifest family resolves. No perturbation target is nominated.

| Fixture | Functional energy | Random-null mean / 95% interval | Exact/JVP error | Fixture-effect label |
| --- | ---: | ---: | ---: | ---: |
| `learned_unused_auxiliary` | 0.000000 | 0.264024 / [0.004496, 0.849129] | 0.000e+00 | 0.000 |
| `causally_effective_choice` | 1.000000 | 0.228624 / [0.000119, 0.789183] | 0.000e+00 | 1.000 |
| `cross_attention_candidate` | 0.000000 | 0.281160 / [0.000241, 0.892555] | 0.000e+00 | 0.500 |
| `adapter_geometry_candidate` | 0.500000 | 0.320530 / [0.000511, 0.911631] | 0.000e+00 | 0.700 |

## Contract and controls

- Orientation: `activation-side basis V[in,k]; declared-output Jacobian J[out,in]; restriction = ||J@V||_F^2 / ||J||_F^2`.
- Exact Jacobian and JVP numerator agree on every fixture; the denominator also has a deterministic 64-probe Hutchinson VJP estimate with 95% interval.
- Every snapshot includes repeated random orthonormal, raw-weight, functional top/middle/bottom, covariance-only, state-permutation, and norm-matched random-checkpoint controls.
- Compiler/legal membership is immutable metadata and is never differentiated through. The tested intervention wrapper fails if it changes.
- Semantic meaning-v2 and protected/debt joins remain unavailable under the current floor gate; they are not synthesized from fixture labels.

## Evidence inventory

- `slm217`: fixture only; no compatible checkpoint plus DecisionEvent manifest
- `slm218`: zero complete provenance-resolvable checkpoint families
- `slm125`: fixture report only; no retained compatible adapter family
- `current_checkpoint_retrospective_eligible`: False

## Decision

- analytic fixtures validate activation-side orientation, exact/JVP parity, deterministic Hutchinson uncertainty, and all preregistered subspace controls
- the learned-unused and causally-effective fixtures separate as designed, but fixtures are not model evidence
- no compatible retained checkpoint plus exact-state manifest resolves for the required retrospective families
- SemanticFloorGateV1 is inconclusive; semantic causal interpretation remains blocked
- no matrix or band is eligible for SLM-220 coupling-based perturbation

Eligible matrices/bands for a later coupling perturbation: **none**.

## Reproduction

```bash
timeout 170s env PYTHONPATH=src .venv/bin/python -m scripts.run_causal_subspace_fixture --check
```
