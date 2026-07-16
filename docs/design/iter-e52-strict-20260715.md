# E52 strict symbol-boundary evaluation — 2026-07-15

Strict constrained LTR evaluation of E52 remained invalid despite the matrix
structural gain. Results were parse 0/3, raw syntax validity 0, structural
similarity 0.3569, fidelity/reward 0, and p50 latency 27.6s.

Failure evidence includes malformed symbol/literal adjacency such as adjacent
quoted symbols and excess `Stack` arguments. No constrained dead ends were
recorded; the decoder completes a candidate, but semantic OpenUI validation
rejects it.

Decision: do not promote E52 based on structural similarity alone. The next
iteration must target semantic arity/closure validity for native symbol
sequences.

This is strict scratch smoke evidence, not a ship claim.
