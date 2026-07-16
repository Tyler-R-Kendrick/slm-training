# E54 symbol boundary weight 1 — 2026-07-15

E54 reduced the symbol-boundary adjacency loss from E52's 2.0 to 1.0 while
keeping the fidelity loss and training recipe fixed.

The interpolation did not recover structure: raw syntax validity was 1/3 and
structural similarity fell to 0.2584, versus E52's 0.4444. Parse,
placeholder fidelity, and reward remain zero.

Decision: reject boundary weight 1. E52's weight-2 setting remains the best
boundary-loss result, but it still does not produce a usable strict parser.

This is scratch smoke evidence, not a ship claim.
