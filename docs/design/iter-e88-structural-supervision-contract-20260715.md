# E88 structural supervision on visible contracts — 2026-07-15

E88 retrained the visible-contract corpus with grammar-LTR-primary supervision,
fidelity loss 4.0, and symbol-boundary loss 2.0 while disabling the explicit
template fast path.

The bounded smoke matrix probe (n=1, 128 steps) reported parse/fidelity 1.0,
structure 0.65, and reward 0.997, but the emitted result was produced by the
existing certified template fallback after learned decode failure. It is not
learned structural evidence.

Decision: reject E88 as a training intervention. Stronger losses did not
survive strict learned decoding; continue using the new fallback counters to
prevent false promotion.

This is bounded scratch evidence, not a ship claim.
