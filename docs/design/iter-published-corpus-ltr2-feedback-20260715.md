# Iteration: published remediated corpus training feedback (2026-07-15)

This verification run used `scripts/train_model.py --train-version remediated`, resolving the source-controlled corpus at `src/slm_training/resources/train_data/remediated` rather than the gitignored generated output directory. The run loaded **585 records** and preserved manifest SHA `928ec8d4921954c7736d2386fe7abf88bbef75523a7cfe404792f45ddcd5d4ba`.

The matched 64-step seed-0 LTR2 recipe reached weighted held-out NLL **7.294** with interval feedback at steps 32 and 64. The bounded one-record smoke probe produced **0.5375 structural similarity**, **0.5 component recall**, **0.2 placeholder validity**, and **3.09 s** p50 latency. Parse rate and reward remained **0**, and AgentV recorded **0 passed / 1 failed**.

This confirms the published corpus is both visible to the Training Data dashboard and consumable by future training runs. The output remains a rejected syntax candidate; this iteration validates data provenance, not a model-quality gain.
