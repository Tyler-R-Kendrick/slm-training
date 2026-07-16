# E48 legal candidate-count telemetry — 2026-07-15

The constrained picker now exposes the size of its legal candidate pool. On
the E48 checkpoint, parse remained 0/3 and dead ends remained 12. The
aggregate last legal-candidate count was 2.667, but this value is not yet a
per-dead-end sample: it can reflect successful picks before a later dead end.

This is useful harness feedback but not sufficient to claim the legal set is
empty or populated at the exact failure. The next telemetry refinement will
persist per-dead-end position and candidate count together, then the model
intervention will be chosen from that evidence.

This is scratch smoke evidence, not a ship claim.
