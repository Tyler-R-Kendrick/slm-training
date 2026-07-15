# Iteration: final-only loss feedback (2026-07-15)

Two identical two-step scratch runs used the remediated corpus (manifest
`928ec8d4921954c7736d2386fe7abf88bbef75523a7cfe404792f45ddcd5d4ba`), batch
size 8, learning rate `6e-4`, seed 0, and CPU scratch TwoTower.

The run with `loss_eval_every=1` spent 79.54% of 26.14 seconds in loss suites
and produced weighted held-out NLL 36.676 at 2,904 target tokens. The control
with cadence 0 completed in 2.90 seconds, but exposed a harness bug: it wrote
no loss feedback at all (`nll_history=[]`, `final_loss_eval=null`).

Cadence 0 now means no intermediate loss checks but always performs one final
loss evaluation when a test directory is available. A one-step verification
produced complete final feedback: weighted NLL 47.999 at 1,632 target tokens,
with persisted `run_insights.json` and `train_telemetry.json`. This keeps the
fast loop while preserving the evidence required for iteration.
