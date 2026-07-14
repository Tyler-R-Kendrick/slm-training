"""Training and data harnesses.

Subpackages (each is a harness with a clear job):

- ``train_data`` / ``test_data`` — versioned corpus build + leakage gates
- ``model_build`` — train / eval loop + ship gates
- ``rl`` — GRPO-lite + trajectory RL (own harness)
- ``preference`` — composite reward + DPO surrogate
- ``distill`` — self-distillation trace store / SFT
- ``quality`` — curriculum, soft rejects, skeleton retrieval
- ``experiments`` — scaling ladders + promotion protocol
- ``annotations`` — human feedback ingestion for train / preference
"""
