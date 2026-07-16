# E86 strict learned contract decoding — 2026-07-15

E86 evaluated the E82 checkpoint with grammar LTR primary/repair, no
unconstrained fallback, no template fast path, and a 64-token cap.

The prediction remained invalid: parse and raw syntax were 0.0, structural
similarity 0.21, reward 0.0, and exact placeholder fidelity 0.75. Latency was
6,692.03 ms p50. Telemetry recorded zero template fast-path/fallback uses and
one constrained dead end, so this is genuine learned constrained output rather
than certified fallback.

Decision: reject E86. The dead-end trace and malformed native token sequence
are the next decoder/training feedback target.

This is a bounded scratch diagnostic, not a ship claim.
