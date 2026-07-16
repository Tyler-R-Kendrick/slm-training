# E52 symbol boundary supervision — 2026-07-15

E52 retained E50's fidelity weight 4.0 and added a targeted loss over native
symbol positions plus their immediate neighbors. This directly supervises
the delimiters and transitions surrounding `<SYM_*>` rather than increasing
the global symbol loss.

The smoke result improved structural similarity to 0.4444, above E50's
0.4083 matrix result, but parse, placeholder fidelity, and reward remained
zero. Predictions have more coherent `Stack` structure and placeholder
strings, but still overproduce arguments and fail closure.

Decision: retain the boundary-loss direction and strict-evaluate E52 next;
inspect delimiter closure errors before another training sweep.

This is scratch smoke evidence, not a ship claim.
