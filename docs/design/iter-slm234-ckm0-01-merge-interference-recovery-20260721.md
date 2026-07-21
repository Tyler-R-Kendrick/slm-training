# SLM-234 (CKM0-01): TIES-vs-average merge signal recovery under synthetic interference (slm234-ckm0-01-merge-interference-recovery-20260721)

**Matrix set:** `slm234_merge_interference_recovery`
**Version:** `ckm0-01-v1`
**Status:** fixture
**Claim class:** wiring
**n_siblings:** 5  **density:** 0.2  **seeds:** [0, 1, 2, 3, 4]
**Disposition:** partial_confirmation_mechanism_specific

TIES robustly beat naive averaging (>= 90% win rate across the seed x conflict_prob grid) on mean_abs_noise_residual, signal_magnitude_recovery, matching the hypothesis for those metrics, but not on cosine_similarity (win_rate=10/20=0.50, worst_gap=-0.1109 at conflict_prob=0.45); signal_sign_recovery_rate (win_rate=12/20=0.60, worst_gap=-0.0388 at conflict_prob=0.45). TIES's magnitude-based trim reliably removes the small-magnitude noise-coordinate residual and preserves a larger mean signed projection at signal coordinates than naive averaging when these are robust, which is the core claimed noise-suppression mechanism working as intended. Where cosine similarity and/or raw sign-recovery rate are not robust, the most likely reading is that TIES's disjoint-merge commits harder (larger magnitude) to whichever sign wins the per-coordinate magnitude-weighted election -- amplifying the result whether that per-coordinate election happens to be right or wrong, which can inflate the merged vector's norm and reduce full-vector cosine similarity even while the signal-restricted mean projection still favors TIES. The strict 'no worse on every metric' hypothesis is falsified; the mechanism-specific claim (trim + magnitude preservation) is supported, the direction-fidelity claim is not.

## Hypothesis

On a synthetic sibling-checkpoint construction where a signal_fraction subset of parameter coordinates carries a large fixed-sign consensus update independently interfered per-sibling with probability conflict_prob (opposite sign, comparable magnitude), and the remaining coordinates carry only small-magnitude sibling-independent noise, the repo's real merge_checkpoints(method='ties') recovers the ground-truth consensus direction (cosine similarity), its magnitude (mean signed projection at signal coordinates), its sign (sign-recovery rate at signal coordinates), and suppresses noise-coordinate residual, at least as well as merge_checkpoints(method='average') across every tested conflict_prob and seed -- reproducing TIES-Merging's two claimed mechanisms (magnitude-based trim of noise, and interference-resistant disjoint-sign merge) on the repo's actual implementation rather than only exercising it.

## Falsifier

For any metric, TIES's mean value across seeds is measurably worse (beyond a small win-margin tolerance) than naive averaging's at any tested conflict_prob, or TIES's win rate for that metric is not near-universal (>= 90%) across the (seed, conflict_prob) grid.

## Honest caveats

- Fixture/wiring evidence only: a synthetic parent + sibling checkpoint construction with a hand-designed ground-truth signal/noise structure, not a measurement on any real trained sibling checkpoints from this repo's training pipeline. No checkpoint promotion, learned merge policy, GPU run, or ship-gate claim is made or implied.
- The signal_fraction (0.2) is deliberately matched to the tested density (0.2) so TIES's magnitude-based trim closely tracks the hand-labeled signal coordinates; a real fine-tuned sibling delta's magnitude spectrum need not separate this cleanly, and a mismatched density/signal_fraction is not explored here.
- conflict_prob is applied i.i.d. per (sibling, signal coordinate); real task interference between sibling checkpoints is unlikely to be i.i.d. Bernoulli and may be structured (e.g. concentrated in specific layers).
- This harness does not modify merge_checkpoints, validate_merge_manifests, or any other harness_core.lineage code -- it only exercises the unmodified merge_checkpoints entry point with method='average' and method='ties'.
- 5 seeds x 4 conflict levels is enough to see whether an effect is consistent, not a formal significance test; no p-values are computed.

## Metric summary (TIES vs average, across all seeds x conflict_probs)

| metric | higher is better | TIES win rate | worst gap | worst gap at conflict_prob |
| --- | --- | --- | --- | --- |
| cosine_similarity | True | 10/20 (0.50) | -0.1109 | 0.45 |
| signal_magnitude_recovery | True | 19/20 (0.95) | -0.0466 | 0.45 |
| signal_sign_recovery_rate | True | 12/20 (0.60) | -0.0388 | 0.45 |
| mean_abs_noise_residual | False | 20/20 (1.00) | 0.0000 | 0.00 |

## Mean results by method x conflict_prob (averaged across seeds)

| conflict_prob | method | cosine_sim | signal_magnitude_recovery | signal_sign_recovery | mean_abs_noise_residual |
| --- | --- | --- | --- | --- | --- |
| 0.00 | average | 0.9986 | 0.9967 | 1.0000 | 0.0191 |
| 0.00 | ties | 0.9994 | 1.0005 | 1.0000 | 0.0018 |
| 0.15 | average | 0.9017 | 0.6972 | 0.9620 | 0.0188 |
| 0.15 | ties | 0.9125 | 0.9166 | 0.9556 | 0.0018 |
| 0.30 | average | 0.7069 | 0.4147 | 0.8369 | 0.0191 |
| 0.30 | ties | 0.6541 | 0.6589 | 0.8274 | 0.0017 |
| 0.45 | average | 0.2004 | 0.0982 | 0.5793 | 0.0188 |
| 0.45 | ties | 0.1523 | 0.1525 | 0.5762 | 0.0017 |

## No-go for promotion

This report is wiring/fixture evidence on a synthetic construction. It does not measure any real sibling checkpoints from this repo's training pipeline, does not change merge_checkpoints or any harness_core.lineage code, and does not authorize automatic merge promotion (merge output is always a new screened challenger per model-lineage.md, unchanged by this harness).

## Reproducibility

```bash
python -m scripts.run_slm234_merge_interference_recovery
```
