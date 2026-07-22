# E930-E931: typed-array slot capacity

E930-E931 bound fresh forward binders in a typed array when every permitted
item family provably requires content and the direct references already equal
the remaining opaque-slot count. The compiler removes the continuation comma
before it would create an unsatisfiable fresh binder; generic component arrays
and slot-free item families are unchanged.

The E891 checkpoint, E842 held-out rows, canvas cap 192, 12-second deadline,
strict compiler-tree policy, plan weights 4/2, and honest opaque slot contract
are fixed. E930 is the same-revision default-off control.

| Run | Treatment | parse | meaning-v1 | strict-v2 | slot fidelity | gold structure | gold type recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E930 | v248 control | 1.0000 | 0.6000 | 0.2000 | 0.6800 | 0.3419 | 0.5810 | 0.8488 | 0 / 2 | 0/1 |
| E931 | typed slot capacity | 0.8000 | 0.8000 | 0.6000 | 0.8000 | 0.5700 | 0.7714 | 0.7640 | 1 / 1 | 0/1 |

The local capacity decision works: the Form path closes `Buttons` after one
button binder rather than reserving an unbounded sequence. The resulting
`b2 = Input(":slot_6")` exposes a separate role error, however: the opaque
slot occupies `Input.name`, producing `placeholder_semantic_role_mismatch`
instead of the valid placeholder/content property. Dual-card still times out.

Retain the conservative capacity check only behind the existing default-off
`compiler_schema_component_types` capability and reject E931. It improves
fidelity, structure, type recall, and fallback count versus E930, but parse and
reward regress, strict-v2 falls from E929's 0.8 to 0.6, and AgentV fails. The
next generalized action is to admit opaque symbols only at official placeholder
properties during schema-constrained decode. No ship gates ran and no
checkpoint was created.

Both evals emitted AgentEvals JSONL and AgentV bundles. Result stamps are dirty
because they include the intended v248 implementation and unrelated concurrent
experiment files; those unrelated files remain untouched.
