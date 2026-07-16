# E151 — Constrained top-k=1 ablation (2026-07-15)

Reducing the grammar candidate top-k to `1` did not recover the constrained decoder. The one-record replay remained unparsable and timed out at 20 seconds. Captured timings were 13,675.8 ms denoiser, 4,825.7 ms picker, and 1,937.1 ms DFA/probe work across 4,959 counted operations. This rules out candidate width as the sole timeout cause.
