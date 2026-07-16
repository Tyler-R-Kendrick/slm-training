# E225 — compiler-tree evaluation invariant

Status: **evaluator fixed; existing E224 checkpoint reevaluated; tree path proven
active; eight ship gates failed; no training or checkpoint promotion**.

E221–E224 scoreboards persisted `compiler_decode_mode=tree`, but decode telemetry
showed zero compiler candidates and all selection traces came from
`ltr_repair`. The model only entered compiler-tree decode when
`grammar_ltr_primary` was independently true. The evaluator therefore claimed a
tree policy while executing MaskGIT followed by legacy constrained repair.

The generalized fix makes any non-`off` compiler mode imply primary constrained
LTR in four places: evaluator config, runtime checkpoint overrides, autoresearch
command compilation, and the generation guard itself. This does not add output
literals or weaken any gate.

The existing E224 checkpoint was reevaluated against canonical
`eval:remediated` with local-only weights, lexer output, schema and slot context,
no DESIGN context, tree mode, and unconstrained fallback disabled. Effective
policy now records `grammar_ltr_primary=true`. Every suite had nonzero compiler
candidates, restricted projections, forced tokens, and `compiler_tree` traces;
full projections and fallback counts remained zero.

| Suite | n | compiler candidates | syntax | meaningful parse | structure | component recall | fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 904 | 0.0000 | 0.0000 | 0.4625 | 0.2500 | 0.0000 |
| held_out | 5 | 1,823 | 0.0000 | 0.0000 | 0.4597 | 0.1567 | 0.0000 |
| adversarial | 4 | 1,133 | 0.0000 | 0.0000 | 0.5784 | 0.4583 | 0.0000 |
| ood | 4 | 1,420 | 0.0000 | 0.0000 | 0.3059 | 0.2083 | 0.0000 |
| rico_held | 3 | 2,733 | 0.0000 | 0.0000 | 0.5729 | 0.4444 | 0.0000 |

Compared with the superseded E224 evaluation, tree decoding materially improves
structure and component recall and reduces failed gates from 12 to eight. It
still fails syntax because generated content properties use schema-invalid fixed
strings instead of required placeholders, and some declaration boundaries are
malformed. AgentV passed 0/5 with no execution errors. This proves the symbolic
tree is now applied and isolates the next decoder defects to schema-conditioned
content choices and parser-state boundary advancement, rather than model
undertraining or sampler policy.
