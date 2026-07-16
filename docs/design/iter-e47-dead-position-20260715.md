# E47 dead-end position diagnostic — 2026-07-15

Dead-end position telemetry was added and evaluated on the BOS-aware E47
checkpoint. Strict constrained LTR remained 0/3 parse with eight dead ends and
no unconstrained fallback. The last dead-end position summed to 5 across the
three examples (mean 1.667), showing that failures occur in the first few
tokens after `root`, rather than after a long partially valid program.

This narrows the next experiment to explicit root-to-assignment prefix
conditioning/supervision. It does not justify relaxing constrained decoding.

This is scratch smoke evidence, not a ship claim.
