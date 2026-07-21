# SLM-227 (NCS2-04): Muon/AdamW convergence-direction sweep (slm227-muon-convergence-20260721)

**Matrix set:** `slm227_muon_convergence`
**Version:** `ncs2-04-v1`
**Status:** fixture
**Claim class:** wiring

## Hypothesis

At matched initialization, learning rate, data, and step budget, the Muon/AdamW hybrid optimizer's final training loss moves in a consistent direction relative to plain AdamW across seeds, once run long enough (more steps and records than SLM-222's 2-step single-record smoke test) for the orthogonalized-momentum update to diverge from AdamW.

## Falsifier

The Muon arm's final loss is not consistently lower (or higher) than AdamW's across the swept seeds (no seed-majority direction), or either arm produces a non-finite loss or non-finite parameters at any seed.

## Honest caveats

- Fixture/wiring evidence only: no trained model, checkpoint promotion, GPU run, or ship-gate claim.
- The full O0-O4 matched AdamW-vs-Muon campaign (capacity- and data-matched, with spectral LR control) still requires local E224+ checkpoints and dedicated GPU time and remains future work.
- The fixture uses a tiny scratch-context model overfitting a handful of synthetic records; a consistent direction here is evidence about this optimizer pair's fixture-scale dynamics, not about downstream OpenUI quality, generalization, or which optimizer to ship.
- Learning rate is matched but not tuned per optimizer; a real campaign would sweep LR separately for each arm before comparing convergence.

## Sweep

- steps per arm: 40
- records: 4
- seeds: [0, 1, 2, 3, 4]

## Per-seed results

| seed | adamw last_loss | muon last_loss | muon - adamw | winner |
| --- | --- | --- | --- | --- |
| 0 | 20.9197 | 45.7798 | +24.8601 | adamw |
| 1 | 19.4246 | 34.9696 | +15.5450 | adamw |
| 2 | 20.5108 | 41.1482 | +20.6374 | adamw |
| 3 | 22.2486 | 45.3956 | +23.1471 | adamw |
| 4 | 25.9424 | 56.7334 | +30.7910 | adamw |

## Summary

- muon_wins: 0
- adamw_wins: 5
- ties: 0
- unstable_seeds: 0
- mean_delta (muon - adamw): 22.996113586425782
- stdev_delta: 5.006776061745488
- all_finite: True

## Disposition

**consistent_adamw_lower_loss**

AdamW's final loss was lower than Muon's in all 5/5 decided seeds (mean delta +22.9961).

## Go / no-go decision

**No-go for promotion.** This report is wiring/fixture evidence only over a tiny scratch-backend model and synthetic overfit data. No checkpoint, GPU train, or ship gate is claimed, and a consistent per-seed loss direction here does not establish that either optimizer is better for real OpenUI training.

## Reproducibility

```bash
python -m scripts.run_slm227_muon_convergence --mode plan-only
python -m scripts.run_slm227_muon_convergence --mode fixture
```

