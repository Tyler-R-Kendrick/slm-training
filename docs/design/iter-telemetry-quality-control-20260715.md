# Accumulation quality control (2026-07-15)

At the same remediated data manifest and 2,904 target tokens, the
`grad_accum=1` baseline reached weighted held-out NLL **36.46** after four
optimizer steps. The `grad_accum=2` control reached **48.87** after two
optimizer steps. Both runs performed exactly one loss-suite evaluation and
persisted run insights.

The result rejects `grad_accum=2` as a default recipe: its lower optimizer
update count does not preserve convergence in this bounded control. It remains
available for future experiments with an explicitly tuned learning rate or
larger training budget.

The same run exposed and fixed duplicate final-step evaluation. A scheduled
loss evaluation was previously repeated by the forced final evaluation, which
inflated telemetry and runtime without adding evidence.
