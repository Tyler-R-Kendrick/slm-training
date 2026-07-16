# E48 singleton-admission bypass — 2026-07-15

The constrained picker now bypasses stream probing when the finalized DFA or
slot-contract admission contains exactly one non-special token. This is safe:
the grammar has already certified the sole candidate, so probing cannot add
information and must not reject it.

Regression tests passed: 28 passed. The matched E48 checkpoint still produced
`root` for all three examples and remained 0/3 parse. The persisted trace shows
the model chose `NL` after `<BIND_0>`, yielding `<bos> <BIND_0> NL`; the eventual
dead end is therefore reached from a multi-token admitted set, not a rejected
singleton. The bypass is retained as a correctness fix, but it does not explain
this failure.

This is scratch smoke evidence, not a ship claim.
