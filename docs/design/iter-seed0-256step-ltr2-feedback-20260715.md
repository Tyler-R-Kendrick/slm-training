# Iteration: 256-step LTR2 feedback (2026-07-15)

This run extended the selected seed-0 compositional LTR2 recipe to 256 steps (`ltr_loss_weight=2.0`, `fidelity_loss_weight=0.5`, batch 8, CPU) with held-out loss feedback every 32 steps. The unchanged remediated corpus contained 585 training records; no DESIGN.md context was used.

Weighted held-out NLL fell monotonically after step 128 and reached **5.614** at step 256. The selected step-256 checkpoint was then evaluated with a bounded one-record smoke probe and AgentV persistence. It produced **0.2292 structural similarity**, **0 component recall**, **0 placeholder validity**, and **3.23 s** p50 latency. Parse rate and reward remained **0**; AgentV recorded **0 passed / 1 failed**.

Reject longer training for this recipe. The lower loss did not translate into executable OpenUI output and structural quality regressed versus the selected 64-step LTR2 control (0.5375 structure, 0.5 component recall, 0.2 placeholder validity). This is evidence for targeting serialization-aligned supervision/data rather than simply increasing steps.
