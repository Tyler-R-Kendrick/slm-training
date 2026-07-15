# Physical batch-size control (2026-07-15)

With the same 8,845-target-token budget, `batch_size=8, grad_accum=1, lr=6e-4`
reached weighted NLL **22.50**, essentially matching tuned `grad_accum=2`
(**22.59**) and remaining slightly worse than the baseline (**21.79**).
Training spans were small relative to the loss-suite checks, which consumed
**70.67%** of wall time.

This supports effective batch size as the main lever rather than accumulation
mechanics alone. Keep batch size 8 explicit and validate generated quality
before any recipe change.
