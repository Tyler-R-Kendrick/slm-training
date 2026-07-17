# E271 — preference optimizer geometry

Date: 2026-07-17
Status: **completed; Adam preconditioning breaks the MGDA certificate**

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

## Final result

The final profile repeated the latest-main clean gate (`0 behind / 2 ahead`)
and separated fresh Adam from AdamW. The negative alignments are effectively
identical:

| Direction | Split/kind | Dot | Cosine |
| --- | --- | ---: | ---: |
| Adam | held-out `grammar_comma` | -1298.05 | -0.009135 |
| AdamW | held-out `grammar_comma` | -1298.03 | -0.009134 |
| Adam | train `grammar_lsqb` | -374.78 | -0.003447 |
| AdamW | train `grammar_lsqb` | -374.87 | -0.003448 |

Decoupled weight decay is therefore negligible. Adam's first-step adaptive
`g/(|g|+eps)` transform itself changes the certified raw MGDA direction enough
to oppose an active held-out grammar objective.

## Decision

Do not add another gradient solver or tune AdamW. The next bounded experiment
is a one-step MGDA plus SGD preflight, because an SGD update is collinear with
the certified combined gradient. It must retain the unchanged optimizer-state
backtracking and held-out per-kind guard, and it must repeat latest-main
reconciliation immediately before training.

Final evidence:
[`quality-matrix-v10-e271-optimizer-geometry-results.json`](quality-matrix-v10-e271-optimizer-geometry-results.json).
