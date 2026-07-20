# E612 — authored typed-array nonempty margin

Date: 2026-07-20
Status: completed, rejected as a quality baseline, not promotable

E612 adds a default-off, compiler-local margin that prevents an authored
prompt-plan component from closing an empty typed array when its item schema
can reach visible string slots and at least one visible slot remains unused.
The highest-scoring legal non-close token is floored two points above the
current legal maximum.

The matched OOD `n=4` replay completed normally. Gallery changes from
`ImageGallery([])` to `ImageGallery([$s45])`, proving the empty close was
displaced, but `$s45` is an unresolved state reference rather than a visible
gallery slot. It therefore still has zero fidelity, validity, component recall,
and reward. Dashboard, modal, and auth are prediction-identical to E611.

Every aggregate quality metric is exactly unchanged from E611: meaningful-v1
0.75, fidelity 0.70, validity 0.72, structure 0.7729, recall 0.6875, reward
0.7148, AST-node F1 0.7579, and AST-edge F1 0.6310. Total emitted tokens rise
85→86. The lower measured latency is not treated as a performance conclusion
from this single small replay.

Reject the highest-scoring-non-close policy as the next quality baseline. Keep
the default-off lever and negative result for reproducibility. The next
iteration should floor the schema-derived typed item start instead of an
arbitrary legal `any` expression, then verify that the required image-property
path reaches a public visible slot.

Strict meaning-v2 remains zero and AgentV remains 0/1, so this is not a
promotion or ship result. No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e612-authored-typed-array-nonempty-20260720.json).
