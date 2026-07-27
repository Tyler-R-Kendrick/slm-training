# Lever: multi-seed selection (s30 + ASAP + t30)

**Honesty:** fixture_or_scratch / smoke n=3. **Not ship.**

## Hypothesis
Selecting the best seed among a small fixed set (42,47,48) with a fixed completed-steps recipe improves reliability vs single seed.

## Results

# Multi-seed selection lever (s30 ASAP t30)
seed=42 parse=0.3333333333333333 meanful=0.0 reward=0.31233333333333335 empty=2 lat=30003.88 stopped=steps steps=30
seed=47 parse=1.0 meanful=0.3333333333333333 reward=0.7653333333333334 empty=0 lat=16281.59 stopped=steps steps=30
seed=48 parse=0.6666666666666666 meanful=0.0 reward=0.0 empty=1 lat=9372.65 stopped=steps steps=30
SELECTED seed=47 run=exp_lever_multiseed_s30_seed47

Captured: 2026-07-27T14:32:42.796708+00:00
