# E47 Silver+ 1024-step surface retrain — 2026-07-15

The controlled E47 run was extended from 256 to 1,024 CPU steps with the same
judged Silver+ corpus, lexer-native target, and doubled LTR loss. Generation
still failed all three smoke examples. Every prediction was exactly `root`.

Teacher-forced loss improved enough to show that structural targets are being
seen (`structural mean NLL 3.78`, legal structural NLL 2.02), but free-running
decoding collapses before the required assignment transition (`root = ...`).

| metric | result |
| --- | ---: |
| parse / language validity | 0/3 |
| structural similarity | 0.000 |
| placeholder fidelity | 0.000 |
| reward | 0.000 |

Decision: reject longer training as a sufficient intervention. The next
experiment should inspect the root-to-assignment conditioning/target path and
add a prefix-level structural adherence metric, rather than treating parse as
the only structural signal.

This is scratch smoke evidence, not a ship claim.
