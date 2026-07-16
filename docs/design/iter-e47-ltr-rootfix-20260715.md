# E47 BOS-aware LTR supervision — 2026-07-15

Lexer-native targets are BOS-less, but the LTR suffix path assumed position 1
was always the first content token. The mask now detects BOS: compositional
targets start at position 1, while lexer-native targets start at position 0.

Focused tests passed: 31 passed, 3 deselected. A matched 256-step E47 retrain
on the judged Silver+ corpus still generated `root` for all three smoke cases:

| metric | result |
| --- | ---: |
| parse | 0/3 |
| structural similarity | 0.000 |
| fidelity/reward | 0.000 |
| latency p50 | 3,990 ms |

The fix remains because it corrects the training contract, but it is not the
sole cause. Next inspect the post-root logits and constrained dead-end/EOS
telemetry before another training sweep.

This is scratch smoke evidence, not a ship claim.
