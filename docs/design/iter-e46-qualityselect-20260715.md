# E46 quality-aware checkpoint selection — 2026-07-15

## Result

Interrupted after the first evaluation, but the run produced a valid
quality-selected checkpoint. The selector marked step 16 as `ship_best` based
on generation quality rather than held-out loss.

| suite | n | parse | placeholder fidelity | structural similarity | reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.000 | 1.000 | 0.6489 | 0.9690 |
| held_out | 5 | 0.600 | 1.000 | 0.5539 | 0.9892 |

Recipe: scratch context, root corpus, E46 lexer/factorized/structural-mask /
template-fill configuration, 128 requested steps, batch size 8, evaluation at
step 16, smoke plus held-out suites, curriculum enabled.

The matrix supervisor progress file remained `running` because the interrupted
wrapper did not finalize after checkpoint evaluation. The checkpoint artifacts
and evaluation JSON remain under
`outputs/runs/iter-e46-qualityselect-20260715/qx_e46_champion/`.

Interpretation: checkpoint selection by generation quality can retain a useful
early checkpoint (smoke parse 1.0) even when the longer run later regresses.
This is diagnostic evidence, not a ship claim: held-out parse is only 0.6 and
constrained generation still needs the next feedback-driven intervention.
