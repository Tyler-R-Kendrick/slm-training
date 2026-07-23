# SLM-183 (PQR): powered cluster-aware confirmation protocol (slm183-power-protocol-20260723)

Matrix set: `slm183_power_protocol`

Version: `pqr-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

A powered confirmation protocol can separate seed variance from target variance and estimate the minimum detectable effect size.

## Falsifier

The protocol collapses seed and target variance into a single pooled estimate and cannot produce a calibrated MDE curve.

## Scenarios

| scenario_id | n_targets | paths_per_target | n_seeds | base_rate | sigma_target | sigma_seed |
| --- | --- | --- | --- | --- | --- | --- |
| mixed_effects_binary | 8 | 2 | 2 | 0.70 | 0.30 | 0.15 |

## Sample cells

| cell_id | target | seed | n | successes | mean | wilson_low | wilson_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target000_seed0 | 0 | 0 | 2 | 2 | 1.000 | 0.342 | 1.000 |
| target000_seed1 | 0 | 1 | 2 | 1 | 0.500 | 0.095 | 0.905 |
| target001_seed0 | 1 | 0 | 2 | 0 | 0.000 | 0.000 | 0.658 |
| target001_seed1 | 1 | 1 | 2 | 1 | 0.500 | 0.095 | 0.905 |
| target002_seed0 | 2 | 0 | 2 | 1 | 0.500 | 0.095 | 0.905 |
| target002_seed1 | 2 | 1 | 2 | 2 | 1.000 | 0.342 | 1.000 |
| target003_seed0 | 3 | 0 | 2 | 2 | 1.000 | 0.342 | 1.000 |
| target003_seed1 | 3 | 1 | 2 | 0 | 0.000 | 0.000 | 0.658 |
| ... | | | | | | | | (8 more cells) |

## Seed variance components

- target variance: **0.0346**
- seed variance: **0.0176**
- pooled variance: **0.0522**

## ICC (one-way)

- ICC: **-0.1455**
- clusters: **8**

## MDE curve

| effect_size | power |
| --- | --- |
| 0.000 | 0.050 |
| 0.020 | 0.090 |
| 0.040 | 0.100 |
| 0.060 | 0.070 |
| 0.080 | 0.110 |
| 0.100 | 0.140 |
| 0.120 | 0.080 |
| 0.150 | 0.100 |

## Conclusions

| conclusion | value | classification |
| --- | --- | --- |
| seed_variance_detected | True | decidable |
| mde_achievable_at_08 | False | decidable |
| cluster_aware_ci_finite | True | decidable |
| holm_rejections | 9 | decidable |

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The power-protocol utilities, cluster bootstrap, ICC, and MDE simulation are wired and exercised on synthetic data, but no real eval records or trained model were used. The protocol remains ``retain_diagnostic`` / ``blocked_pending_real_eval`` until it is run on actual suite results.

## Honest caveats

- Synthetic outcomes are generated from a mixed-effects logit model; real eval records may have different correlation structures.
- MDE simulation uses a normal approximation (no scipy dependency); small-n results are exploratory only.
- No model is trained; this is wiring evidence for the protocol only.
- Cluster bootstrap and ICC assume a single target_cluster_id level; nested clusters are not modeled here.

## Reproducibility

```bash
python -m scripts.run_flow_power_protocol --mode plan-only
python -m scripts.run_flow_power_protocol --mode fixture
python -m scripts.run_flow_power_protocol --mode analyze-existing --iter-json <path>
```

## Exact command

```bash
python -m scripts.run_flow_power_protocol --mode fixture --output-dir /tmp/pytest-of-codex/pytest-276/test_fixture_writes_design_doc1 --n-targets 8 --paths-per-target 2 --n-seeds 2 --seeds 0,1,2,3,4
```
