# E48 root assignment commit invariant — 2026-07-15

The end-to-end LTR repair path was committing `NL` after the native root
binding even though the valid program requires `root = ...`. A commit-boundary
invariant now substitutes the assignment token when the current prefix is a
bare root binding. This protects the state machine from a broad/stale picker
admission while retaining normal constrained selection elsewhere.

Focused tests passed: 33 passed, 3 deselected. On the same E48 checkpoint,
predictions advanced from `root` to `root =` for all three smoke examples.
Parse remains 0/3 because the first RHS transition still dead-ends at position
3. This is a real structural improvement, not a ship result.

Next target: post-assignment component selection and its training signal.
