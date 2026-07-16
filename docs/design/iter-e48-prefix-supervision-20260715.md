# E48 prefix structural supervision — 2026-07-15

E48 added an explicit `ltr_prefix_loss_weight=4.0` to overweight the first
three content transitions, targeting the `root` → `=` failure. It used the
same judged Silver+ corpus and 256-step scratch recipe as E47.

The result was unchanged: all three predictions were `root`, with parse,
structural similarity, fidelity, and reward all zero. Structural teacher-forced
NLL was 3.984 (legal structural NLL 2.224).

Decision: reject prefix-loss weighting alone. The knob is retained because it
expresses the correct hypothesis, but the next intervention must inspect the
actual post-root constrained logits/dead-end candidate set.

This is scratch smoke evidence, not a ship claim.
