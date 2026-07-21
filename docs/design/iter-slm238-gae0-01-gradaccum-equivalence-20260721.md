# SLM-238 (GAE0-01): gradient-accumulation equivalence sweep (slm238-gae0-01-gradaccum-equivalence-20260721)

**Matrix set:** `slm238_gradaccum_equivalence`
**Version:** `gae0-01-v1`
**Status:** fixture
**Claim class:** wiring

## Hypothesis

At matched initialization, learning rate, optimizer, and total record set, training the real model_build train loop with grad_accum_steps=2 (micro-batch 4) reaches a final training loss within 15% relative difference of direct training with batch_size=8, grad_accum=1 (same effective batch size), across multiple seeds -- i.e. gradient accumulation is a close, unbiased numerical stand-in for a larger physical batch at fixture scale, not a systematically different training regime, and the grad_accum / effective_batch_size telemetry fields correctly report the accounting.

## Falsifier

The accum arm's final loss differs from the direct arm's by more than 15% relative in a consistent direction across a majority of seeds (a systematic bias, not stochastic mask-draw noise), or the accel.grad_accum / accel.effective_batch_size telemetry fields do not match the configured values, or either arm produces a non-finite loss.

## Honest caveats

- Fixture/wiring evidence only: a tiny scratch-context TwoTower model overfitting 8 synthetic records for 40 optimizer steps says nothing about gradient-accumulation behavior on a real training corpus or at production model scale.
- The two arms are not expected to be bit-identical: mask/corruption randomness is drawn per forward call from the global RNG stream, so an 8-record single call and two sequential 4-record calls consume that stream differently even from matched initial state. This experiment measures whether the resulting loss trajectories stay close, not whether they are numerically identical.
- Learning rate, weight decay, and optimizer (AdamW) are matched but not independently retuned per arm; a production comparison would also check gradient-norm distributions and multi-epoch generalization, not just final training loss on a fixture that only ever overfits.
- The CLOSE_RELATIVE_TOLERANCE=15% threshold was chosen before running the sweep and is a fixture-scale judgment call, not a derived statistical bound.

## Sweep

- optimizer steps per arm: 40
- records: 8
- close-relative-difference tolerance: 15%
- seeds: [0, 1, 2, 3, 4]

## Per-seed results

| seed | direct last_loss | accum last_loss | delta (accum-direct) | rel diff | close? | winner |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 19.2314 | 19.7263 | +0.4949 | 0.0257 | True | direct |
| 1 | 19.5163 | 18.8768 | -0.6395 | 0.0328 | True | accum |
| 2 | 21.9957 | 21.6378 | -0.3579 | 0.0163 | True | accum |
| 3 | 16.6605 | 16.3508 | -0.3097 | 0.0186 | True | accum |
| 4 | 13.8394 | 13.8969 | +0.0574 | 0.0042 | True | direct |

## Summary

- accum_wins: 3
- direct_wins: 2
- ties: 0
- unstable_seeds: 0
- close_seeds (within tolerance): 5/5
- mean_relative_diff: 0.01950319234048945
- mean_delta (accum - direct): -0.15095939636230468
- stdev_delta: 0.3917429505120172
- all_finite: True
- all_metadata_ok (accel.grad_accum / accel.effective_batch_size match config): True

## Disposition

**close_approximation_confirmed**

All 5/5 decided seeds stayed within the pre-registered 15% relative-difference tolerance (mean relative diff 0.0195, mean delta -0.1510). Gradient accumulation behaves as a close, unbiased stand-in for the equivalent physical batch at this fixture scale.

## Go / no-go decision

**No-go for promotion; positive mechanism characterization.** This report is wiring/fixture evidence only over a tiny scratch-backend model and synthetic overfit data. No checkpoint, GPU train, or ship gate is claimed. A `close_approximation_confirmed` disposition supports treating grad_accum_steps as a faithful physical-batch stand-in at fixture scale for future scaling-ladder or memory-constrained runs; any other disposition means that substitution should not be assumed without a matched-batch control.

## Reproducibility

```bash
python -m scripts.run_slm238_gradaccum_equivalence --mode plan-only
python -m scripts.run_slm238_gradaccum_equivalence --mode fixture
```

