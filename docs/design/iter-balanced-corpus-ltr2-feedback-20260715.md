# Iteration: target-balanced corpus feedback (2026-07-15)

The published `remediated_unique` corpus contains **198 records**, one per unique OpenUI target, versus 585 records and 198 unique targets in `remediated` (473 parent-derived augmentations). The 64-step seed-0 LTR2 run consumed the source-controlled `remediated_unique` path and preserved interval held-out telemetry.

Held-out weighted NLL was **6.945** at step 64. Full smoke feedback (`n=3`) produced **0.1325 structural similarity**, **0 component recall**, **0 placeholder validity**, and **1.56 s** p50 latency. Parse rate and reward were **0**, with AgentV **0 passed / 1 failed**.

Reject `remediated_unique` as a training default. Target-only deduplication reduced useful prompt/layout coverage; keep `remediated` as the control and design a diversity-balanced corpus using both prompt families and structural targets before the next data intervention.
