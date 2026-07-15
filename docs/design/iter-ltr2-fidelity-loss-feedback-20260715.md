# Iteration: LTR2 fidelity-loss weight feedback (2026-07-15)

This iteration tested stronger placeholder-token supervision on the unchanged 585-record remediated corpus. The 64-step seed-0 recipe used `ltr_loss_weight=2.0`, `fidelity_loss_weight=2.0`, interval held-out loss feedback every 32 steps, and no DESIGN.md context. The bounded smoke probe evaluated one record and persisted AgentEvals JSONL plus an AgentV bundle.

Held-out weighted NLL was **7.504**. Smoke output reached **0.3417 structural similarity**, **0.25 component recall**, and **0.0667 placeholder validity** at **3.44 s** p50 latency. Parse rate and reward were **0**, with AgentV **0 passed / 1 failed**. The matched LTR2 control was 0.5375 structural similarity, 0.5 component recall, 0.2 placeholder validity, and the same zero parse/reward rates.

Reject the stronger fidelity weight. The earlier `fidelity_loss_weight=0.5` run was not a distinct intervention because 0.5 is already the model default; that control mistake is recorded so future comparisons specify every recipe knob explicitly. The train harness now persists loss weights, schema/retrieval settings, and honesty mode in `train_summary.json`.
