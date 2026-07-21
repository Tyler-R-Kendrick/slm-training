# SLM-191 (FFE2-03): termination-policy fixture matrix (slm191-termination-matrix-20260721)

Matrix set: `slm191_termination_matrix`

Version: `ffe2-03-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU, no model, no checkpoint, and no ship-gate claim is made.

## Hypothesis

Termination semantics materially change the empirical endpoint distribution, edit-count distribution, and premature/late-stop rates on exact CTMC fixtures; a shared TerminationPolicy protocol lets direct-policy and flow samplers be compared on the same scalar signals.

## Falsifier

All six termination arms produce identical endpoint distributions and edit-count distributions on every exact domain, or the oracle-length arm is not the strongest baseline, or explicit-stop/absorbing-hazard arms are uncalibrated (Brier/ECE far above chance) on the known target signal.

## Arms

k_value: 4
n_samples_per_arm: 100

| arm_name | n_samples | stop_rate | absorb_rate | mean_edit_count | p95_edit_count | brier_stop | ece_stop | endpoint_tv_vs_exact | reached_target_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| explicit_stop | 100 | 1.000 | 0.000 | 1.39 | 2.0 | 1.000 | 1.000 | 0.500 | 0.000 |
| absorbing_hazard | 100 | 1.000 | 1.000 | 1.51 | 2.0 | 0.000 | 0.000 | 0.500 | 0.000 |
| fixed_k | 100 | 1.000 | 0.000 | 1.40 | 2.0 | 0.000 | 0.000 | 0.500 | 0.000 |
| fixed_k_plus_selector | 100 | 1.000 | 0.000 | 1.40 | 2.0 | 0.000 | 0.000 | 0.500 | 0.000 |
| hybrid_min_progress | 100 | 1.000 | 0.000 | 1.32 | 2.0 | 1.000 | 1.000 | 0.500 | 0.000 |
| oracle_length | 100 | 1.000 | 0.000 | 0.00 | 0.0 | 0.000 | 0.000 | 0.500 | 0.000 |
| explicit_stop | 100 | 1.000 | 0.000 | 2.00 | 2.0 | 0.250 | 0.500 | 0.844 | 0.000 |
| absorbing_hazard | 100 | 0.330 | 0.280 | 7.46 | 9.0 | 0.000 | 0.000 | 0.232 | 0.130 |
| fixed_k | 100 | 1.000 | 0.000 | 4.00 | 4.0 | 0.000 | 0.000 | 0.844 | 0.000 |
| fixed_k_plus_selector | 100 | 1.000 | 0.000 | 4.00 | 4.0 | 0.000 | 0.000 | 0.844 | 0.000 |
| hybrid_min_progress | 100 | 1.000 | 0.000 | 2.00 | 2.0 | 0.250 | 0.500 | 0.844 | 0.000 |
| oracle_length | 100 | 0.330 | 0.000 | 7.30 | 9.0 | 0.000 | 0.000 | 0.259 | 0.130 |
| explicit_stop | 100 | 1.000 | 0.000 | 1.09 | 2.0 | 1.000 | 1.000 | 0.500 | 0.000 |
| absorbing_hazard | 100 | 1.000 | 1.000 | 1.05 | 1.0 | 0.000 | 0.000 | 0.500 | 0.000 |
| fixed_k | 100 | 1.000 | 0.000 | 1.05 | 1.0 | 0.000 | 0.000 | 0.500 | 0.000 |
| fixed_k_plus_selector | 100 | 1.000 | 0.000 | 1.10 | 2.0 | 0.000 | 0.000 | 0.500 | 0.000 |
| hybrid_min_progress | 100 | 1.000 | 0.000 | 1.04 | 1.0 | 1.000 | 1.000 | 0.500 | 0.000 |
| oracle_length | 100 | 1.000 | 0.000 | 1.09 | 2.0 | 0.000 | 0.000 | 0.500 | 0.000 |

## Target rows

Total target rows: 3

| domain | n_terminals | exact_edit_count_support | oracle_edit_count |
| --- | --- | --- | --- |
| toy_layout | 0 | 4 | 0 |
| choice_sequence | 12 | 5 | n/a |
| canonical_edit_graph | 0 | 3 | 2 |

## Disposition

**supports_termination_diversity**

Arms produce differentiated edit counts and/or endpoint distributions on the exact CTMC fixture.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The TerminationPolicy protocol, six reference arms, and calibration instrumentation are exercised over deterministic exact CTMC domains with synthetic model signals. Production samplers must replace the synthetic signals with learned STOP, total-hazard, absorption, and selector heads and re-run on real checkpoints before any ship claim.

## Honest caveats

- Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.
- STOP score, absorption probability, and selector probability are synthetic signals derived from the exact CTMC (hazard, absorption probabilities, edit-distance to target); production samplers must replace them with model heads.
- Only the canonical_edit_graph domain has a known target program; oracle_length and selector-based arms are intentionally weaker on toy_layout and choice_sequence.
- Exact edit-count and holding-time distributions are empirical samples, not closed-form CTMC jump-time distributions.
- Domains are intentionally tiny (<= a few hundred states) so the matrix stays CPU-only.

## Reproducibility

```bash
python -m scripts.run_termination_matrix --describe
python -m scripts.run_termination_matrix --exact-fixture
```
