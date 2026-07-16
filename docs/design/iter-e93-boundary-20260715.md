# E93 boundary guard diagnostic (2026-07-15)

E93 tested a narrow grammar guard intended to reject a binder immediately after
the closed `root = Stack(...)` expression when no newline had been emitted.
The checkpoint and strict decode configuration were held constant from E91;
only this grammar rule changed.

The guard did not solve the failure. On the one-case smoke diagnostic, parse
and raw syntax stayed at `0.0`, while structural similarity fell from `0.5333`
to `0.4958`. Contract precision/recall remained `0.8/1.0`, and the decode
still produced repeated `b3 = ...` repair statements. The run had one
constrained dead end at position 1, with zero certified or unconstrained
fallbacks.

Decision: reject this guard and keep the grammar focused on the existing root
RHS invariant. The next investigation should address candidate probing and
the dead-end recovery path, not add another statement-boundary exclusion.
