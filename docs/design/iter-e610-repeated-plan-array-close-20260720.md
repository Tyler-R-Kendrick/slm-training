# E610 — repeated-plan-family array closure

Date: 2026-07-20
Status: completed, retained as a research lever, not promotable

E610 adds a default-off, compiler-state-derived close margin for nested arrays
owned by a prompt-plan family with multiple required instances. After one
array item is complete, the legal close token is floored two points above the
best legal alternative. Singleton families and the terminal root are
unchanged. The choice state now counts completed variadic items directly;
existing reference counts retain their original meaning.

The capped, matched OOD `n=4` replay completes normally. Dashboard retains
Button, Callout, and both Card bindings in its verified root while shrinking
from 163 to 60 output symbols. Its reward recovers from 0 to 0.865.
Modal and auth are prediction-identical to E609.

Aggregate meaningful-v1 improves from 0.50 to 0.75, structure from 0.6667 to
0.7646, reward from 0.4835 to 0.6998, and p95 latency falls from 29.61 s to
12.98 s. Fidelity and validity fall slightly versus E609 (0.70 to 0.65 and
0.72 to 0.69), so the preregistered strict no-regression condition is not met.
However, E610 exceeds E608 on fidelity, validity, structure, recall, reward,
and latency while matching its 0.75 meaningful-v1 rate. Retain the lever as
the next scratch baseline.

Strict meaning-v2 remains zero. Dashboard assigns the first metric placeholder
to both Cards and still misuses Callout schema/value roles; gallery remains an
empty `ImageGallery`. The next iteration should allocate distinct visible
slots across repeated planned instances before addressing gallery family
selection.

No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e610-repeated-plan-array-close-20260720.json).
