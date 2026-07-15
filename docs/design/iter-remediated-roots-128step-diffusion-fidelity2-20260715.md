# Diffusion plus stronger fidelity loss — 2026-07-15

This matched run increases fidelity loss weight from 0.5 to 2.0 while keeping
the source-controlled `remediated_roots` corpus, diffusion masking, TwoTower
scratch context, 128 steps, batch size 8, and LTR loss weight 2.0 unchanged.

- 108 records / 94 unique targets
- 60,846 target tokens; telemetry total 33.27 seconds
- Best held-out weighted NLL: 4.953764
- Broad mean NLL: 5.514317

The best checkpoint was evaluated with explicit constrained decoding, LTR
repair, 64-token cap, one attempt, and smoke `n=3`. Parse rate, structural
similarity, and reward were all 0.0; there were no decode timeouts and p50
latency was 2,172 ms.

Reject the checkpoint. Stronger fidelity loss did not improve either loss or
constrained generation. The next intervention must change target composition
or the output head/objective more fundamentally.
