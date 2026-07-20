# E614 — typed-object choice state

Date: 2026-07-20
Status: completed, retained as the scratch research baseline, not promotable

E614 fixes the generalized grammar-state defect exposed by E613. When a choice
decoder enters an object required by an active schema, the frame now retains
the declared properties, their value schemas, required keys, seen keys, and
whether additional properties are legal. Closed objects reject unknown and
repeated keys, cannot close before required properties are present, and use
schema-derived values for bounded completion. Variadic item/reference counts
also enter the decode-state cache signature.

The authoritative matched OOD `n=4` replay completed normally in about 30
seconds with no decode
timeout or fallback. Gallery contracts from E613's recursive 160-token object
to:

```openui
root = Stack([v0], "column")
v0 = ImageGallery([{alt: ":ood.gallery.alt", src: ":ood.gallery.alt", details: ":ood.gallery.caption"}])
```

Against E613, aggregate structure recovers 0.7452→0.7729, AST-node F1
0.7365→0.7579, fidelity improves 0.7417→0.7833, validity improves
0.8450→0.8700, reward improves 0.8865→0.9020, emitted tokens fall 233→93,
and p95 latency falls 19.20→12.60 seconds. Component recall returns from the
spurious recursive-component value of 0.7500 to the E611 baseline of 0.6875.

Against E611, structure, recall, and AST scores are unchanged while fidelity
improves 0.7000→0.7833, validity 0.7200→0.8700, and reward
0.7148→0.9020. The cost is eight emitted tokens, +0.23 seconds p50, and
+1.14 seconds p95. Retain the generalized grammar repair and use the E614
policy as the next scratch research baseline.

Strict meaning-v2 remains zero and AgentV remains 0/1. Gallery still binds
`src` to the `alt` slot and omits four required prompt placeholders, so this is
not a promotion or ship result. The next iteration should expose the active
typed-object property to schema-role slot scoring so `src`, `alt`, and
`details` select their matching public slots.

No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e614-typed-object-state-20260720.json).
