# E51 symbol fidelity weight 8 — 2026-07-15

E51 doubled E50's native symbol fidelity loss weight from 4.0 to 8.0. The
matched 256-step Silver+ scratch run regressed: parse, structural similarity,
placeholder fidelity, and reward all returned zero, with p50 latency 9.96s.

E50's weight 4.0 remains the better bounded setting (strict structural
similarity 0.4244). Reject further scalar fidelity-weight increases. The next
experiment should add explicit native symbol sequence/closure telemetry or a
targeted loss rather than continue this sweep.

This is scratch smoke evidence, not a ship claim.
