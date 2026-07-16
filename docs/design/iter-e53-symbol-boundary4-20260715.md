# E53 symbol boundary weight 4 — 2026-07-15

E53 increased only the symbol-boundary adjacency loss from 2.0 to 4.0 while
keeping E52's fidelity weight and recipe fixed.

Raw syntax validity improved to 2/3, but structural similarity fell to 0.3094
from E52's 0.4444 matrix result. Parse, placeholder fidelity, and reward remain
zero.

Decision: reject boundary weight 4. The narrower boundary objective improves
surface closure at the cost of broader component structure; retain E52's
weight-2 setting for further semantic diagnostics.

This is scratch smoke evidence, not a ship claim.
