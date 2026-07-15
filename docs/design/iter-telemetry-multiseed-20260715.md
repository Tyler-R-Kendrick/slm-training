# Longer multi-seed telemetry comparison (2026-07-15)

Two six-step runs with the same tuned accumulation recipe and data manifest
persisted 48 per-example token-loss proxies each. Seed 0 averaged **37.05**;
seed 1 averaged **46.11**. Their top-five high-loss IDs had zero overlap, and
source-family means shifted between seeds.

This does not support deleting or reweighting records. The dominant uncertainty
is recipe/model stochasticity at this small budget, so the next experiment is a
larger multi-seed recipe comparison with generated-quality checks.
