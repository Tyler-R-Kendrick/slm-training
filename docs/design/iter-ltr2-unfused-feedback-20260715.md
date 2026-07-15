# Iteration: LTR2 unfused supervision feedback (2026-07-15)

This 64-step seed-0 run held the LTR2 recipe constant and disabled fused mask+LTR loss with `--no-fuse-ltr`, forcing the existing second denoiser forward for left-to-right supervision. It used the remediated 585-record corpus, `ltr_loss_weight=2.0`, explicit `fidelity_loss_weight=0.5`, interval loss feedback every 32 steps, and no DESIGN.md context.

Held-out weighted NLL was **7.263**. The bounded smoke probe produced **0.2333 structural similarity**, **0.25 component recall**, and **0 placeholder validity** at **3.11 s** p50 latency. Parse rate and reward were **0**, and AgentV recorded **0 passed / 1 failed**.

Reject unfused supervision: it regressed all executable-output diagnostics against the fused LTR2 control (0.5375 structure, 0.5 component recall, 0.2 placeholder validity, parse/reward 0). The recipe telemetry now also persists `fuse_ltr_loss` so future comparisons cannot omit this distinction.
