# Iteration: seed-2 early loss feedback (2026-07-15)

A two-step scratch TwoTower run used the remediated corpus, batch size 8,
learning rate `6e-4`, seed 2, and final-only loss feedback. It consumed 2,526
target tokens and reached complete weighted held-out NLL **36.915**.

The run persisted `train_summary.json`, `train_telemetry.json`,
`run_insights.json`, `nll_history.jsonl`, and `loss_suites.json`. Telemetry
records effective batch size 8. This early point remains seed-sensitive relative
to prior seeds, so no data reweighting or recipe change is justified from this
short run.
