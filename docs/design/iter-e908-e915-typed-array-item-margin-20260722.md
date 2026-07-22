# E908-E915: lexer typed-array item margin

E908 tests a hard rejection of unresolved forward binders in schema-typed
arrays. It doubles strict-v2 from 0.2 to 0.4 but collapses three complex rows to
certified minimal fallbacks, so the hard rule is rejected. E909-E912 then expose
two no-effect activation bugs while extending the existing typed-item margin to
lexer compiler-tree decode. E913 fixes the real empty-array forest guard; E914
is its same-revision margin-0 control. E915 limits schema-forced items to closed
leaf components whose required public-schema properties do not recursively
require another component.

| Run | Arm | parse | meaning-v1 | strict-v2 | fidelity | structure | recall | reward | fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E906 | no slot context, margin 0 | 1.0000 | 0.8000 | 0.2000 | 0.8400 | 0.3492 | 0.7810 | 0.9178 | 1 | 0/1 |
| E908 | hard untyped-forward rejection | 1.0000 | 0.4000 | 0.4000 | 0.5019 | 0.3406 | 0.5619 | 0.7744 | 5 | 0/1 |
| E909/E910 | activation attempts | 1.0000 | 0.8000 | 0.2000 | 0.8400 | 0.3492 | 0.7810 | 0.9178 | 1 | 0/2 |
| E914 | slot-context, margin 0 control | 1.0000 | 0.4000 | 0.2000 | 0.5133 | 0.2945 | 0.4476 | 0.7826 | 4 | 0/1 |
| E913 | v241 typed-item margin 2 | 1.0000 | 0.4000 | 0.4000 | 0.5019 | 0.3406 | 0.5619 | 0.7744 | 5 | 0/1 |
| E915 | v242 closed-leaf margin 2 | 1.0000 | 0.4000 | 0.4000 | 0.5019 | 0.3406 | 0.5619 | 0.7744 | 5 | 0/1 |

The treatment makes the compact settings row strictly valid and gives it both
fidelity and component recall 1.0. The input row also remains strict with both
metrics 1.0. The low aggregate fidelity/recall is not a penalty for
minification: form, dual-card, and tabs collapse to `TextContent(":slot_0")`
and fail strict-v2 because required placeholders and prompt components are
missing.

Retain the canonical lexer implementation and capability declaration as an
opt-in research lever, default off. Reject margin 2 for E891 and do not run
smoke, promote, sync, or serve it. E906 remains the best E891 diagnostic recipe.
No ship gates ran, and AgentV is 0/8 across E908-E915.

The result stamps are dirty solely because another session modified and created
unrelated SLM experiment files in this shared worktree. Those files remain
untouched and are outside this evidence.
