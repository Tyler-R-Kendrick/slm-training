# E47 constrained dead-end telemetry — 2026-07-15

The decoder now persists `constrained_dead_ends` alongside timing, token, and
fallback counters. On the BOS-aware 256-step E47 checkpoint, strict constrained
LTR produced 0/3 parse and recorded 8 dead ends total (2.667 per example),
with no unconstrained fallback.

This distinguishes a clean parser failure from a repair loop that pads after
exhausting legal candidates. The repeated dead ends confirm the next target is
the model's post-root logits/conditioning, not wider candidate search.

Focused tests passed: 15 passed, 3 deselected. This is scratch smoke evidence,
not a ship claim.
