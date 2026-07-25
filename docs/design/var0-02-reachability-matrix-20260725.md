# VAR0-02 (SLM-423): per-variant x suite edit-reachability matrix

- generated_at: `2026-07-25T19:54:05Z`
- seed: `root = Stack([], "column")`
- max_edits: 8, node_budget: 120
- verdict policy: reachable_fraction is computed over decided cases only; UNKNOWN_BUDGET cases are never counted as unreachable; a suite with no corpus is corpus_unavailable, never zero-reachable; a variant with no enumeration engine wired in this issue is NOT_MEASURED, never inferred from another variant's row and never a fabricated number. Reachability is a space-coverage proof, never a model-quality claim.

> `reachable_fraction` is a property of a *(variant, suite)* pair, not of the program. A number measured for one variant's action alphabet may never be cited as a verdict on any other variant, or on patch-based generation in general.

## Matrix

| variant | train | smoke | held_out | adversarial | ood | rico |
| --- | --- | --- | --- | --- | --- | --- |
| `repl_operators` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| `tree_edit_diffusion` | corpus_unavailable | 0.0 (n=3, unk=0) | 0.0 (n=5, unk=0) | 0.0 (n=3, unk=1) | 0.0 (n=4, unk=0) | 0.0 (n=35, unk=0) |
| `twotower_prompt_ast` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

## Alphabet fingerprints

- `repl_operators`: `647e09b2259d5a132c3a925755f7d6d9c01f66b9ef67e491159a2631b614154d`
- `tree_edit_diffusion`: `ab2662a497d8359ffaee46ebbd4bee3789f5b0f2accaf8bf46c5dee489622dab`
- `twotower_prompt_ast`: `a429e7ddd2ec3b781fa7c9e7b70ca470437aa694d7d3833f76c3791a00ab149d`

## NOT_MEASURED reasons

- `repl_operators`: successor enumeration for this variant requires dsl/operators/legal_set.py's enumerate_operator_legal_set, which operates over OperatorStateV1 / ReferenceTableV1 / ApplicationProvenanceV1 built from the live operator registry -- a different state representation and search than TreeEditSpace's statement list. Wiring a seed->target BFS over that alphabet (build an OperatorStateV1 per corpus record, enumerate legal operator actions, apply each to get a successor source, compare canonical forms against the target) is a new enumeration engine, which this issue's explicit scope excludes ('changing any action alphabet... measurement and attribution only'; 'if it is not measured in this issue, it is NOT_MEASURED'). Recorded here rather than left blank or inferred from the tree-edit row.
- `twotower_prompt_ast`: this variant has no edit-action alphabet to measure: it is a full-AST prompt->AST denoiser over structural tokens, not a seed->target edit-transition system, so 'edit reachability' as this analyzer defines it (bounded BFS over an action alphabet from a minimal seed) does not apply to it. Recorded as NOT_MEASURED rather than a fabricated number or a blank cell.

## Reason-code histograms (measured rows only)

- **tree_edit_diffusion**:
  - smoke: `{"unsupported_component": 3}`
  - held_out: `{"unsupported_component": 5}`
  - adversarial: `{"budget": 1, "needs_direction_change": 2, "unsupported_component": 1}`
  - ood: `{"unsupported_component": 4}`
  - rico: `{"needs_direction_change": 35}`
