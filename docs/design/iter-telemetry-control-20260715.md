# Gradient accumulation control (2026-07-15)

Matched CPU scratch controls used the same remediated data manifest and six
microbatches. `grad_accum=1` completed six optimizer steps in 43.05s; the
`grad_accum=2` control completed three optimizer steps in 30.76s. Backward
remained dominant (64.52% vs 69.27%). Both runs persisted `run_insights.json`.

The harness now reports the mean unscaled loss over an accumulation window;
previously it reported only the final microbatch, which made accumulation
feedback misleading. The timing result supports keeping the control available,
but this bounded run is not quality or convergence evidence and does not justify
changing the default recipe.
