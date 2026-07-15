# Gradient accumulation probe (2026-07-15)

A one-step CPU scratch run with batch size 4 and `grad_accum=2` persisted its
checkpoint, telemetry, and run insights. Backward remained the bottleneck at
69.84%, but two microbatches completed in 22.35s (~11.18s each), versus 18.74s
for the one-microbatch baseline. This is a performance signal only; quality
and convergence still need a controlled multi-step comparison.
