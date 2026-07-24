# Null-calibrated spectral learning disposition

**Schema:** `SpectralDispositionV1`

**Report hash:** `1882a4937d3a7351a9945d29e0ba1887e5722f7cf1d17969e352cb1758b35d7c`

**Evidence cutoff:** `0b7bc033953abc7858e5eb7620a1c480d792cfd5`

**Semantic floor:** `inconclusive` (`7839ef6b6e37710d487757da9170017d7b76a9d12ca1fb314bdb0fa23a4dd83d`)

## Executive finding

The program adopts four fail-closed diagnostics and no training or production mechanism. Raw alpha, verifier-conditioned spectral LR, causal spectral retention, and WW-PGD/trace-log correction are rejected or blocked in the measured scope; negative evidence remains replayable.

This is an evidence synthesis, not a new train/eval/profile run. It writes no checkpoint and makes no AgentV, promotion, or ship claim.

## Primary sources and fidelity boundary

- [Yang et al., Heavy-Tailed Self-Regularization in Deep Neural Networks (JMLR 2021)](https://jmlr.org/papers/v22/20-410.html)
- [Martin and Mahoney, Implicit self-regularization in deep neural networks (Nature Communications 2021)](https://www.nature.com/articles/s41467-021-24025-8)
- [WeightWatcher spectral RG draft](https://weightwatcher.ai/rg_theory_webpage/rg_theory.html)

Repository rows label each use as faithful, adapted, or surrogate. The sources motivate measurements and hypotheses; they do not supply repository checkpoint, causal, protected-objective, or ship evidence.

## Mechanism table

| Category | Mechanism | Issues | Fidelity | Disposition | Default | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| measurement | `native_spectral_snapshot_and_null_cache` | SLM-214 | adapted | **adopt_diagnostic** | `diagnostic_only` | `docs/design/iter-slm214-spectral-snapshot-20260721.json` |
| measurement | `raw_alpha_as_quality_or_criticality_signal` | SLM-214, SLM-215, SLM-226 | surrogate | **reject** | `forbidden` | `docs/design/iter-slm214-spectral-snapshot-20260721.json`, `docs/design/iter-slm215-spectral-atlas-20260721.json`, `docs/design/iter-slm226-absolute-spectral-gate-20260723.json` |
| measurement | `checkpoint_atlas_outcome_prediction` | SLM-215 | adapted | **inconclusive** | `off` | `docs/design/iter-slm215-spectral-atlas-20260721.json` |
| measurement | `fixed_token_spectral_regime_diagnostics` | SLM-216 | adapted | **retain_research** | `diagnostic_only` | `docs/design/iter-slm216-spectral-regime-20260723.json` |
| functional_causal | `decision_conditioned_functional_spectra` | SLM-217 | adapted | **inconclusive** | `off` | `docs/design/iter-slm217-functional-spectra-20260723.json` |
| functional_causal | `cross_attention_and_parent_child_retention_geometry` | SLM-218 | adapted | **inconclusive** | `off` | `docs/design/iter-slm218-cross-attention-retention-20260723.json` |
| measurement | `weightwatcher_stable_rank_parity` | SLM-219 | faithful | **adopt_diagnostic** | `optional_diagnostic` | `docs/design/iter-slm219-correlation-trap-20260723.json` |
| functional_causal | `correlation_trap_early_warning` | SLM-219 | adapted | **reject** | `not_supported` | `docs/design/iter-slm219-correlation-trap-20260723.json` |
| functional_causal | `activation_side_causal_restriction_estimator` | SLM-220 | adapted | **adopt_diagnostic** | `diagnostic_only` | `docs/design/iter-slm220-causal-subspace-20260723.json` |
| functional_causal | `isospectral_and_band_causal_hypothesis` | SLM-221 | adapted | **blocked** | `blocked` | `docs/design/iter-slm220-causal-subspace-20260723.json` |
| optimization_retention | `muon_hybrid_optimizer` | SLM-222 | surrogate | **retain_research** | `off` | `docs/design/iter-slm222-muon-baseline-20260721.json` |
| optimization_retention | `relative_spectral_optimizer_tournament` | SLM-223 | adapted | **blocked** | `blocked` | `docs/design/iter-slm216-spectral-regime-20260723.json`, `docs/design/iter-slm220-causal-subspace-20260723.json`, `docs/design/iter-slm222-muon-baseline-20260721.json` |
| optimization_retention | `verifier_conditioned_spectral_lr` | SLM-224 | adapted | **reject** | `not_applicable` | `docs/design/iter-slm216-spectral-regime-20260723.json`, `docs/design/iter-slm220-causal-subspace-20260723.json` |
| optimization_retention | `causal_spectral_elastic_retention` | SLM-225 | adapted | **blocked** | `not_supported` | `docs/design/iter-slm218-cross-attention-retention-20260723.json`, `docs/design/iter-slm220-causal-subspace-20260723.json` |
| absolute_targeting | `absolute_spectral_target_gate` | SLM-226 | adapted | **adopt_diagnostic** | `fail_closed` | `docs/design/iter-slm226-absolute-spectral-gate-20260723.json` |
| absolute_targeting | `ww_pgd_trace_log_projection` | SLM-227 | adapted | **blocked** | `not_authorized` | `docs/design/iter-slm226-absolute-spectral-gate-20260723.json`, `docs/design/iter-slm220-causal-subspace-20260723.json`, `docs/design/iter-slm222-muon-baseline-20260721.json` |

## Disposition counts

- `adopt_diagnostic`: 4
- `retain_research`: 2
- `reject`: 3
- `blocked`: 4
- `inconclusive`: 3

## Finite-size and raw-alpha boundary

The 200-draw Gaussian-null mean alpha changes from `2.273991` at `128x128` to `3.468501` at `256x128` and `4.973008` at `512x128`. Raw alpha and alpha near 2 are therefore rejected as quality, criticality, champion, or promotion evidence.

## Functional and causal geometry

Functional spectra, cross-attention/retention geometry, and correlation traps remain fixture or research diagnostics. The activation-side restriction estimator is adopted only as an analytic diagnostic. No eligible model band or frozen causal target exists.

## Optimization and retention

Muon is wiring-only; the relative-control tournament was blocked; verifier-conditioned allocation and causal spectral retention were rejected or blocked because their prerequisite mechanism/target did not exist. No spectral training default changes.

## Quality, protected, and cost frontier

No spectral training campaign cleared its activation gates, so there is no qualified quality/protected/cost Pareto frontier. Muon has only two-step wiring evidence; all SVD, guard, checkpoint, and target-hardware cost claims remain unavailable rather than inferred.

## Absolute targeting

AbsoluteSpectralTargetGateV1 is adopted as a fail-closed diagnostic gate with zero authorized roles/shapes. WW-PGD, trace-log projection, and absolute alpha targeting remain `not_authorized`.

## Canonical policy

Keep SpectralSnapshotV1, the analytic activation-side estimator, and AbsoluteSpectralTargetGateV1 as diagnostic/governance owners. Spectral observables cannot influence champion selection, promotion, or production configuration unless a future versioned disposition reaches adopt_primary.

## Evidence identities and non-independence

- `docs/design/iter-slm214-spectral-snapshot-20260721.json` — artifact `37a207da1b5f1669a06939478dda3f194590cdef470e164bdae39ecb8b25f7d7`, file SHA-256 `37a207da1b5f1669a06939478dda3f194590cdef470e164bdae39ecb8b25f7d7`
- `docs/design/iter-slm215-spectral-atlas-20260721.json` — artifact `f3184747850d3b85f97f09117fca00e2c306a70689ae960fc5e5b5e2cc3ef514`, file SHA-256 `f3184747850d3b85f97f09117fca00e2c306a70689ae960fc5e5b5e2cc3ef514`
- `docs/design/iter-slm226-absolute-spectral-gate-20260723.json` — artifact `c413d33274206d2e68552c351a2b0bd959b4ffd1969f1d23e7478cc78ba0fd2e`, file SHA-256 `7b4823a7015589f85fadf0aa375215ddd1acb8d48e4a08b3fc4e657d9946cd65`
- `docs/design/iter-slm216-spectral-regime-20260723.json` — artifact `7fd9f53499195a196080a24748451ced1c5eea89fb52c3a1519e2f6ae1e88675`, file SHA-256 `ca0c254124fe6c04ca3415152bc9e72e4f76ef27a6aed1313945abeecfada220`
- `docs/design/iter-slm217-functional-spectra-20260723.json` — artifact `d9953911c42303bb860db40e326a88612ed9bf17c4b822b028404ac263cd1391`, file SHA-256 `17f9dea5f8f488c9ee8d53d11dd873374761022678d2b032b28a759847663398`
- `docs/design/iter-slm218-cross-attention-retention-20260723.json` — artifact `04fa873a3615b0f695e0bea745bd968516092d5f2ac51ff13e93ba466cf14a72`, file SHA-256 `8e259fc48a4850b6d8656851f6914c390ba9e3be6a5d18a75a116d72af214b65`
- `docs/design/iter-slm219-correlation-trap-20260723.json` — artifact `0340c57d223df0304bb520da638c56b7af786668620f38ac490ca4574d36fffa`, file SHA-256 `b98f5d71a09290ca52c0bf2be8674960ec30dc21a0a49d99d88fdad1d32df6fe`
- `docs/design/iter-slm220-causal-subspace-20260723.json` — artifact `5de0f767ff3844b1074c17c3f0b60e6a38ee45ec0ceea40e0ec84576392a5312`, file SHA-256 `170d070da73e1e4fcdd8a7ebd2d94b4869e75d09a03e01e440a623bb0df13ae1`
- `docs/design/iter-slm222-muon-baseline-20260721.json` — artifact `5b5601362df6bd6e5bcd62578fa596f6056549675dd79015ad0211ab7989b50e`, file SHA-256 `5b5601362df6bd6e5bcd62578fa596f6056549675dd79015ad0211ab7989b50e`

SLM-214 is the shared spectral-statistics owner for SLM-215–220 and SLM-226; those rows are not independent replications. SLM-217 feeds SLM-218/220, while SLM-216, SLM-220, and SLM-222 jointly gate SLM-223–227. Downstream closures therefore preserve prerequisite missingness instead of counting it as repeated negative experiments.

## Per-mechanism rationale and actions

### `native_spectral_snapshot_and_null_cache`

The canonical owner validates deterministic native statistics, synthetic controls, null keys, role classification, and tied-storage deduplication. It makes no quality or ship claim.

Required action: Retain as the canonical inspection API; require null evidence beside raw alpha.

Known confounds: fixture model only; no WeightWatcher production dependency.

### `raw_alpha_as_quality_or_criticality_signal`

The 128x128, 256x128, and 512x128 Gaussian-null means are strongly shape-dependent; proximity to 2 is not positive evidence.

Required action: Reject raw-alpha promotion features and require null-calibrated observables.

Known confounds: finite size; aspect ratio; initializer; tail support; estimator choice.

### `checkpoint_atlas_outcome_prediction`

The atlas wiring works, but committed compatible checkpoint/outcome coverage is insufficient for prediction.

Required action: Revisit only after a provenance-complete checkpoint family exists.

Known confounds: missing checkpoints; non-independent snapshots; semantic floor.

### `fixed_token_spectral_regime_diagnostics`

The measured scratch matrix is useful negative-boundary evidence but is inconclusive and blocks optimizer, semantic, promotion, and ship claims.

Required action: Keep the report replayable; do not route it into training defaults.

Known confounds: optimizer-step count; scratch model; no durable checkpoint.

### `decision_conditioned_functional_spectra`

Orientation and streaming covariance plumbing are validated only on fixtures; no compatible durable checkpoint plus DecisionEvent manifest exists.

Required action: Retain the diagnostic owner but make no predictive or causal inference.

Known confounds: fixture-only; missing exact-state checkpoint family.

### `cross_attention_and_parent_child_retention_geometry`

Zero complete provenance-resolvable checkpoint families were available; no cross-attention role or retention target was nominated.

Required action: Do not select retention targets from synthetic magnitudes.

Known confounds: missing parents; synthetic geometry.

### `weightwatcher_stable_rank_parity`

WeightWatcher 0.7.5 completed 18 comparisons with maximum stable-rank absolute error 8.527e-14. Its fitted alpha remains descriptive only.

Required action: Keep WeightWatcher pinned and optional; use native metrics as the canonical owner.

Known confounds: optional dependency; alpha parity is not quality evidence.

### `correlation_trap_early_warning`

One seed/family and one transient collapse provide no independent non-collapse false-positive denominator; the report explicitly does not support an operational early-stop rule.

Required action: Retain the report for research replay; forbid operational early stopping or promotion decisions.

Known confounds: small trajectory; proxy collapse labels.

### `activation_side_causal_restriction_estimator`

Analytic exact/JVP/Hutchinson contracts pass, but the current-model retrospective is rejected and nominates no eligible bands.

Required action: Keep the estimator; require a provenance-complete family before model use.

Known confounds: analytic labels are not model evidence; semantic floor.

### `isospectral_and_band_causal_hypothesis`

No eligible perturbation bands or provenance-resolvable frozen checkpoint/state family exists, so the battery could not start.

Required action: Reopen only after the causal estimator nominates nonempty frozen targets.

Known confounds: no target artifact; no source checkpoint.

### `muon_hybrid_optimizer`

Only a two-step fixture validates partitioning; no matched convergence, protected, downstream, or GPU evidence exists.

Required action: Keep optional wiring default-off; do not call it the strongest baseline.

Known confounds: single synthetic record; unmatched downstream evidence.

### `relative_spectral_optimizer_tournament`

The regime gate is inconclusive, no causal mechanism exists, and no qualified AdamW/Muon control exists; the tournament was not authorized.

Required action: Do not add a controller until a future versioned gate authorizes exact roles.

Known confounds: no durable checkpoint; no matched baseline.

### `verifier_conditioned_spectral_lr`

SLM-223 supplied no winning spectral direction or controller state, so layering verifier policy was explicitly not applicable.

Required action: No implementation; reopen only after qualified spectral ordering.

Known confounds: missing base mechanism.

### `causal_spectral_elastic_retention`

SLM-221 produced no eligible causal target or frozen weights; implementing a penalty would select from synthetic magnitude or outcomes.

Required action: Do not add retention code without a future causal result.

Known confounds: no target artifact.

### `absolute_spectral_target_gate`

The 200-draw width study is descriptive-only, authorizes no roles/shapes, and its guard blocks every absolute intervention.

Required action: Retain the gate and require it before any absolute-target manifest.

Known confounds: scratch linear probes; no production width boundary.

### `ww_pgd_trace_log_projection`

AbsoluteSpectralTargetGateV1 is descriptive-only with empty authorization and SLM-221 has no causal target; projection is not authorized.

Required action: No projection implementation; the absolute gate remains authoritative.

Known confounds: finite size; norm changes; SVD overhead; guard selection.

## Reproduction

```bash
timeout 170s env PYTHONPATH=src /home/codex/repos/slm-training/.venv/bin/python -m scripts.publish_spectral_disposition --check
```
