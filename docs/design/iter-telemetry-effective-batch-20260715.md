# Iteration: effective-batch telemetry (2026-07-15)

At the same seed, learning rate `6e-4`, remediated corpus, final-only loss
feedback, and 6,252 target tokens, physical batch 8 reached weighted held-out
NLL **27.977**. Batch 4 with gradient accumulation 2 reached **28.065**.
The close result supports effective batch size as the relevant comparison, not
an accumulation-specific quality change.

The training telemetry now records `batch_size`, `grad_accum`, and derived
`effective_batch_size` in both the telemetry metadata and the run summary.
This makes future accumulation/physical-batch comparisons directly auditable.

Both are scratch diagnostics; neither is a ship claim.
