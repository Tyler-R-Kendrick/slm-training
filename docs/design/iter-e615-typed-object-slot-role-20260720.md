# E615 — typed-object slot roles

Date: 2026-07-20
Status: completed, retained as the scratch research baseline, not promotable

E615 extends the existing public-schema role policy through E614's typed object
frames. Schema role discovery now includes inline object and array-item fields
without following component references. Slot suffixes map to compatible public
property names, and the decode bias requires both the owning component and the
active typed-object property to match. This is a general inline-record rule,
not an `ImageGallery` special case.

The clean matched OOD `n=4` replay completed normally in 26.8 seconds with no
decode timeout or fallback. Gallery now emits the intended property bindings:

```openui
root = Stack([v0], "column")
v0 = ImageGallery([{alt: ":ood.gallery.alt", src: ":ood.gallery.img", details: ":ood.gallery.caption"}])
```

Against E614, Gallery fidelity improves 0.3333→0.5000, validity
0.6000→0.7000, and reward 0.7490→0.7990. Aggregate fidelity improves
0.7833→0.8250, validity 0.8700→0.8950, and reward 0.9020→0.9145.
Meaningful-v1, structure, recall, AST F1, and emitted tokens are unchanged.
Measured p50 falls 4.48→4.22 seconds and p95 falls 12.60→10.22 seconds.
Dashboard, modal, and auth predictions remain unchanged. Retain E615 as the
next scratch research baseline.

Strict meaning-v2 remains zero and AgentV remains 0/1. Gallery still omits the
hint and CTA requirements, and its binding report incorrectly labels the
inline object key `alt` as an unresolved reference even though the official
parser and whole-program verifier pass. The next iteration should repair that
general object-key binding-analysis defect before using strict-v2 diagnostics
to target remaining generation failures.

No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e615-typed-object-slot-role-20260720.json).
