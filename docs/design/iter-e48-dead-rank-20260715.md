# E48 post-prefix dead-rank diagnostic — 2026-07-15

The E48 checkpoint was rerun with telemetry for the rank of a singleton
grammar-forced token at each constrained dead end. It remained 0/3 parse and
recorded 12 dead ends, with mean final dead-end position 3.0. The forced-token
rank was unavailable (`-1`) for all failures, meaning these are not simple
misses where one punctuation token was the only legal choice.

The decoder should not force arbitrary punctuation. The next diagnostic must
capture the full legal candidate set and its scores at the first failing
prefix, then target the model or grammar state based on that evidence.

This is scratch smoke evidence, not a ship claim.
