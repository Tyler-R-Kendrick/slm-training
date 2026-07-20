# E613 — schema-derived typed-array item

Date: 2026-07-20
Status: completed, promising direction but rejected as a baseline

E613 keeps E612's authored typed-array nonempty gate but floors the
schema-derived item start instead of the highest-scoring generic expression.
For `ImageGallery.images`, this selects the object opener.

The matched OOD `n=4` replay completed normally. Gallery now consumes the
public `:ood.gallery.caption` slot and its reward rises 0→0.687. Aggregate
fidelity improves 0.7000→0.7417, validity 0.7200→0.8450, component recall
0.6875→0.7500, and reward 0.7148→0.8865. Dashboard, modal, and auth remain
prediction-identical.

The object frame does not retain the array item schema, however, so Gallery
fills arbitrary keys and nested components until the 160-token canvas:
`ImageGallery([{min: Card([...]), justify: Button(...)}])`. Aggregate structure
falls 0.7729→0.7452, AST-node F1 falls 0.7579→0.7365, emitted tokens rise
86→233, and p95 latency rises 9.35→19.20 seconds.

Retain the default-off schema-derived target as a research direction, but keep
E611 as the scratch baseline. The next iteration should propagate typed object
property schemas into the choice state, require known keys, and make required
property closure explicit before testing another decode margin.

Strict meaning-v2 remains zero and AgentV remains 0/1, so this is not a
promotion or ship result. No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e613-schema-derived-typed-item-20260720.json).
