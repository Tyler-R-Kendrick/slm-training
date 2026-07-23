# SLM-217: decision-conditioned functional spectra

**Verdict:** `inconclusive`

**Report hash:** `d9953911c42303bb860db40e326a88612ed9bf17c4b822b028404ac263cd1391`

**Semantic floor:** `7839ef6b6e37710d487757da9170017d7b76a9d12ca1fb314bdb0fa23a4dd83d` (`inconclusive`)

**Orientation:** PyTorch nn.Linear weight is [out_features,in_features]; row activations [n,in_features] map as X @ W.T; functional operator is W @ Sigma^(1/2).

| Kind / role / split | Support / groups | Covariance rank | Raw→functional ESD | Isotropic-null ESD | Permutation-null mean | Eligibility |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `component` / `component_slot` / `held_out` | 8 / 4 | 4 | 1.029645 | 1.056862 | 0.936520 | `eligible` |
| `binding` / `binding_slot` / `held_out` | 8 / 4 | 4 | 1.071854 | 0.773791 | 0.841405 | `eligible` |

## Verdict rationale

- analytical fixture confirms functional spectra diverge across exact-state activation strata
- no compatible durable current-contract checkpoint and DecisionEvent manifest pair is committed
- SemanticFloorGateV1 is inconclusive; semantic interpretation is blocked

This run used deterministic analytical activations and no model checkpoint. It validates the contract and null plumbing only; it does not establish out-of-family predictive value, causal relevance, semantic capability, promotion eligibility, or ship readiness.

## Reproduction

```bash
timeout 170s env PYTHONPATH=src .venv/bin/python -m scripts.run_functional_spectral_fixture --check
```
