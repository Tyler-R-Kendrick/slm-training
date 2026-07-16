# E50 native symbol fidelity supervision — 2026-07-15

E50 increased `fidelity_loss_weight` from 1.0 to 4.0 while retaining lexer
symbol-table substitution, judged Silver+ data, and the decoder repairs. This
was matched to E47/E48 and run for 256 CPU steps.

Structural recovery improved to 0.4083, but parse remained 0/3 and placeholder
fidelity remained 0. The predictions contain more recognizable `Stack`,
`TextContent`, and placeholder forms, but still repeat symbols and malformed
arguments.

Decision: retain the stronger fidelity signal as promising but insufficient.
Next inspect native symbol sequence/closure behavior before further weighting.

This is scratch smoke evidence, not a ship claim.
