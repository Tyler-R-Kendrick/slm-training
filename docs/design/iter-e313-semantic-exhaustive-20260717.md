# E313 semantic-exhaustive choice alignment — 2026-07-17

Status: launch failed before checkpoint; harness repair in progress.

E313 adds the existing semantic-exhaustive compiler-alignment loss to E311's
matched E307 v4 / CPU scratch / 20k-target-token recipe. It trains the actual
denoiser logits at gold compiler-legal root and bound decisions without adding
model parameters.

The first launch stopped after step 7 / 336 target tokens. A gold alignment
token was absent from the compiler decision's candidate tuple, and the harness
raised `ValueError: tuple.index(x): x not in tuple`. No checkpoint was written,
so there is no model-card or bucket update. The last completed batch had 20
alignment rows (2 root, 5 bound) and alignment loss 22.1860.

This is a harness failure, not model evidence. The repair is to skip and
explicitly count gold-outside-candidate rows, then relaunch the unchanged
recipe. This record will be extended with the repaired run's result.
