# CAP2-02 — Mixed-radix FSQ, LFQ, VQ, and continuous latent controls (2026-07-18)

Implementation wiring for Linear SLM-87. Adds a common `LatentCodec` interface and
plugs five codec families into the existing CAP2-01 strict bottleneck harness.
Evidence: [iter-cap2-02-latent-codec-matrix-20260718.json](iter-cap2-02-latent-codec-matrix-20260718.json).

## What was added

- `src/slm_training/models/latent_codec.py` — shared protocol, spec, encoding,
  storage estimate, and diagnostics types.
- `src/slm_training/models/uniform_scalar_codec.py` — uniform K-ary scalar
  baseline (matches `KaryBottleneck` behavior through the new interface).
- `src/slm_training/models/mixed_radix_fsq.py` — mixed-radix finite scalar
  quantization with a deterministic level-vector allocator.
- `src/slm_training/models/binary_lfq.py` — binary lookup-free sign code.
- `src/slm_training/models/learned_vq.py` — learned codebook VQ with commitment
  loss and dead-code diagnostics.
- `src/slm_training/models/continuous_latent.py` — continuous control with
  explicit noise/rate policy; no discrete capacity claim.
- `src/slm_training/models/latent_codec_trainer.py` — generic trainer/evaluator
  that works for any `LatentCodec` plus a small decoder.
- `src/slm_training/harnesses/experiments/cap2_bottleneck.py` — extended with
  `BottleneckArm.codec`, new latent-codec arms, and `evaluate_latent_codec_arm`.
- `tests/test_models/test_latent_codec.py` — round-trip, capacity, and
  convergence regression tests for every codec family.
- `tests/test_harnesses/experiments/test_cap2_bottleneck.py` — matrix inclusion
  and per-codec arm tests.

## Fixture matrix

Run:

```bash
python -m scripts.run_cap2_bottleneck --state-count 41 --seeds 0 \
  --out-dir outputs/runs/cap2_bottleneck
```

Recipe: CPU; state count M=41; single seed; wiring-only honesty mode.

| arm | codec | capacity | exact_rate | occupied | leakage |
| --- | --- | ---: | ---: | ---: | ---: |
| b2d5 | kary/injective | 32 | 0.7805 | 32 | False |
| b2d6 | kary/injective | 64 | 1.0000 | 41 | False |
| t3d3 | kary/injective | 27 | 0.6585 | 27 | False |
| t3d4 | kary/injective | 81 | 1.0000 | 41 | False |
| k2d6 | kary/injective | 64 | 1.0000 | 41 | False |
| k4d3 | kary/injective | 64 | 1.0000 | 41 | False |
| k8d2 | kary/injective | 64 | 1.0000 | 41 | False |
| k7d4_robust | kary/robust | 2401 | 1.0000 | 41 | False |
| k3d7_robust | kary/robust | 2187 | 1.0000 | 41 | False |
| direct_one_hot | direct | 41 | 1.0000 | 41 | False |
| learned_b2d6 | kary/learned | 64 | 1.0000 | 41 | False |
| learned_t3d4 | kary/learned | 81 | 1.0000 | 41 | False |
| fsq_2_3_3_4_5 | mixed_radix_fsq | 360 | 1.0000 | 41 | False |
| lfq_d6 | binary_lfq | 64 | 1.0000 | 41 | False |
| vq_64_d8 | learned_vq | 64 | 1.0000 | 41 | False |
| continuous_d6 | continuous | 6* | 1.0000 | 41 | False |
| uniform_b2d6 | uniform_scalar | 64 | 1.0000 | 41 | False |

\* `continuous_d6` reports latent_dim=6 as its nominal "capacity" for table
alignment, but it is not a discrete bottleneck and is excluded from leakage
classification.

## Hard gates

- Below-capacity discrete arms: 2 (`b2d5`, `t3d3`).
- Leakage violations: 0.
- All discrete above-capacity arms reached 100% fixture reconstruction.

## Honest caveats

- This is a deterministic fixture run, not a production-quality claim.
- `continuous_d6` memorizes 41 states with a 6-D continuous latent and a
  learned decoder; it is included only as a rate/noise-policy control, not as
  evidence that 6 continuous dimensions encode 41 discrete states under the
  strict bottleneck contract.
- No comparison of trainability at scale, wall time, or generalization is made.
- Model-card/README updates are not required: no checkpoint was created or
  promoted.
