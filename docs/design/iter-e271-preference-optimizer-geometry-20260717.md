# E271 — preference optimizer geometry

Date: 2026-07-17
Status: **in progress; fresh AdamW reverses a certified held-out direction**

E271 extends the frozen E270 profile with the exact first-step direction of the
fresh AdamW optimizer used by local preference training. The analytic transform
includes Adam's bias-corrected sign-like preconditioning (`g/(|g|+eps)`) and
AdamW's default decoupled weight decay; no optimizer or checkpoint is mutated.

The initial profile fetched/rebased latest `origin/main`, was clean, and proved
`0 behind / 1 ahead` at `5182694`. The raw MGDA direction remains positively
aligned with every active held-out FTPO-loss gradient, but fresh AdamW reverses
held-out `grammar_comma` (`dot=-1298.03`, cosine `-0.0091`) and train-only
`grammar_lsqb` (`dot=-374.87`, cosine `-0.0034`). This directly explains why
E269's optimizer-consistent scales can regress despite a raw-gradient
common-descent certificate.

The final profile separates Adam's adaptive preconditioning from decoupled
weight decay before choosing a solver-compatible optimizer.

Initial AdamW evidence:
[`quality-matrix-v10-e271-adamw-geometry-results.json`](quality-matrix-v10-e271-adamw-geometry-results.json).
