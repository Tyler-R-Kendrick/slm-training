# Iteration: LTR2 lexer-tokenizer feedback (2026-07-15)

This 64-step seed-0 run kept the LTR2 recipe (`ltr_loss_weight=2.0`, `fidelity_loss_weight=0.5`) and changed only the output tokenizer from `compositional` to `lexer`. It used the 585-record remediated corpus, interval held-out loss feedback every 32 steps, and no DESIGN.md context.

The train summary reached weighted held-out NLL **7.252** at step 64. After fixing an evaluation-harness gap that prevented selecting the lexer tokenizer, the bounded one-record smoke probe produced **0.2625 structural similarity**, **0.5 component recall**, **0 placeholder validity**, and **2.86 s** p50 latency. Parse rate and reward were **0**, with AgentV **0 passed / 1 failed**.

Reject the lexer tokenizer for this branch. It did not repair serialization and reduced structural similarity relative to the compositional LTR2 control. The evaluation CLI now accepts `--output-tokenizer`, so future tokenizer experiments are evaluated against the intended checkpoint configuration.
