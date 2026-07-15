# Iteration: LTR2 structural auxiliary loss feedback (2026-07-15)

A 64-step seed-0 LTR2 run tested `fastpath_aux_weight=0.1` on the unchanged 585-record remediated corpus, with interval held-out loss feedback every 32 steps and a bounded one-record AgentV smoke probe.

Held-out weighted NLL was **7.286**. The smoke probe produced **0.4208 structural similarity**, **0.25 component recall**, and **0.2 placeholder validity** at **3.21 s** p50 latency. Parse rate and reward were **0**, and AgentV recorded 0 passed / 1 failed.

The result is rejected: it did not improve executable syntax over the selected LTR2 control (0.5375 structural similarity, 0.5 component recall, 0.2 placeholder validity, parse/reward 0). Keep the auxiliary loss disabled in the next branch and target token serialization directly. Train summary, telemetry, scoreboard, AgentEvals JSONL, and AgentV artifacts are persisted under the run directories.
