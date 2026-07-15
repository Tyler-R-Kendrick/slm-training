# Iteration: LTR2 diffusion masking feedback (2026-07-15)

This 64-step seed-0 run changed only the training corruption policy from random masking to `mask_pattern=diffusion`, keeping the compositional tokenizer, LTR2 weights (`ltr_loss_weight=2.0`, `fidelity_loss_weight=0.5`), batch size 8, and no DESIGN.md context. Held-out loss feedback ran every 32 steps on the unchanged 585-record remediated corpus.

Weighted held-out NLL reached **7.231**. The bounded one-record smoke probe produced **0 structural similarity**, **0 component recall**, and **0 placeholder validity** at **2.94 s** p50 latency. Parse rate and reward were **0**; AgentV recorded **0 passed / 1 failed**.

Reject diffusion masking for this branch. It did not improve serialization and was worse than the matched random-mask LTR2 control. The train summary recipe now persists `mask_pattern` and `diffusion_policies` so corruption-policy comparisons remain reproducible.
