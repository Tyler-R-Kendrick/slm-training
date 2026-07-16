# E75 archived checkpoint recheck — 2026-07-15

The archived E75 checkpoint was evaluated through the current gold-free
`evaluate_model.py` path with the persisted training data and current lexer
decoder settings.

On the smoke suite (n=3), it scored parse 0.0, exact placeholder fidelity
0.0, structural similarity 0.1287, and reward 0.0. This contradicts the
archived E75 result that reported a full multi-suite pass with fidelity 1.0.

Decision: invalidate the archived E75 promotion evidence. Its earlier result
was produced by a stale or otherwise non-reproducible evaluation path. Only
E76–E78 corrected current-path results are admissible for subsequent research.

This is a reproducibility audit, not a ship claim.
