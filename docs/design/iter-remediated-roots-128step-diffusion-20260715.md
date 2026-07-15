# Diffusion-mask root-target diagnostic — 2026-07-15

The persisted `remediated_roots` corpus was trained with the diffusion mask
schedule to test whether denoising-aligned corruption improves generation.

- TwoTower, scratch context, CPU, compositional output tokenizer
- 108 records / 94 unique targets
- 128 steps, batch 8, LTR loss weight 2.0
- 60,846 target tokens; telemetry total 32.09 seconds
- Best held-out weighted NLL: 4.839584
- Broad mean NLL: 5.381541

The best checkpoint used explicit grammar-constrained decoding, LTR-primary
repair, a 64-token cap, one attempt, and smoke `n=3`. Parse rate, raw syntax
validity, structural similarity, component recall, and reward were all 0.0.
There were no decode timeouts and p50 latency was 1,793 ms.

This is a rejected quality candidate. Diffusion masking improves loss and
latency but does not produce valid constrained OpenUI; the next intervention
must address the output objective or decoder contract.
