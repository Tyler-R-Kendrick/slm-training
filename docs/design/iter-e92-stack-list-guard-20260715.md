# E92 Stack-list semantic guard — 2026-07-15

E92 tested rejecting literals, native symbols, and nested punctuation inside
open `Stack([...])` child lists. Strict learned smoke structure regressed from
E91's 0.5333 to 0.1217; parse and fidelity fell to 0.0, while latency was
5,783.81 ms with no timeout.

Decision: reject and remove the Stack-list guard. It over-constrained a
frontier whose model state was already incomplete. Retain only the narrower
root-RHS invariant and continue from E91's less-regressed baseline.

This is a bounded scratch diagnostic, not a ship claim.
