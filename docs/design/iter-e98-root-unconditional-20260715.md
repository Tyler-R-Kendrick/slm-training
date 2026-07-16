# E98 unconditional root invariant bypass (2026-07-15)

E98 moved the empty-prefix root binding bypass ahead of the lossy DFA token
map. The lexer map exposed broad terminals without `<BIND_0>`, which caused
the prior implementation to miss the known semantic singleton.

The strict E91 smoke diagnostic now records `root_invariant_bypass_count=2`
and `constrained_dead_ends=0` (E97: `0` and `1`, respectively). Parse and raw
syntax remain `0.0`; structural similarity remains `0.5333`; contract
precision/recall remain `0.8/1.0`. Latency increased to `9205.65 ms`, so this
fix removes the false dead end but does not yet repair the malformed learned
sequence.

Decision: retain the bypass and telemetry. The next iteration must address
the post-root repair sequence (`b3` repetition and missing statement
separators), not root admission.
