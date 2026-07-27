# VAR0-02 (SLM-423): per-variant edit-space reachability matrix

- generated_at: `2026-07-26T18:52:40Z`
- seed: `root = Stack([], "column")`
- mode: `extended`, max_edits: 8, node_budget: 120
- reachable_fraction is computed over decided cases only; UNKNOWN_BUDGET cases are reported separately and are never counted as unreachable; suites without a corpus are corpus_unavailable, never zero-reachable. Reachability is a space-coverage proof, never a model-quality claim, and is scoped to one (variant, suite) cell -- never cited as a program-wide status (decode-invariants.md I12/I14).

> Reachability is space coverage, not model quality, and is scoped
> to one (variant, suite) cell. A `reachable_fraction=0.0` for one
> variant says nothing about any other registered variant.

## `repl_operators`

- alphabet_fingerprint: `647e09b2259d5a132c3a925755f7d6d9c01f66b9ef67e491159a2631b614154d`
- seed_id: `repl_operators.caller_supplied_state`

| suite | status | reachable_fraction | decided | unknown_budget |
| --- | --- | --- | --- | --- |
| adversarial | not_measured_deferred | None | None | None |
| held_out | not_measured_deferred | None | None | None |
| ood | not_measured_deferred | None | None | None |
| rico | not_measured_deferred | None | None | None |
| smoke | not_measured_deferred | None | None | None |
| train | not_measured_deferred | None | None | None |

> Enumeration is feasible with existing primitives -- dsl/operators/legal_set.py::enumerate_operator_legal_set plus registry.py::OperatorLibraryV1.apply and registry.py::OperatorStateV1.from_source already compose exactly this pipeline in harnesses/train_data/operator_corpus.py:494-660 (source -> state -> context/library -> legal set -> per-action apply -> successor source, re-validated via validate_with_pack_authority at every step). A standalone enumerate_operator_successors() BFS engine analogous to _enumerate_children is out of this issue's scope (measurement and attribution only, per its own non-goals: 'changing any action alphabet' and building a new search engine are not the same thing, but a new engine is still new capability work) and is deferred to a follow-up issue.

## `tree_edit_diffusion`

- alphabet_fingerprint: `ab2662a497d8359ffaee46ebbd4bee3789f5b0f2accaf8bf46c5dee489622dab`
- seed_id: `tree_edit_diffusion.minimal_valid_program_seed`

| suite | status | reachable_fraction | decided | unknown_budget |
| --- | --- | --- | --- | --- |
| adversarial | measured | 0.0 | 3 | 1 |
| held_out | measured | 0.0 | 5 | 0 |
| ood | measured | 0.0 | 4 | 0 |
| rico | measured | 0.0 | 35 | 0 |
| smoke | measured | 0.0 | 3 | 0 |
| train | corpus_unavailable | None | None | None |

Reason-code histograms:
- **adversarial**: `{"budget": 1, "needs_direction_change": 2, "unsupported_component": 1}`
- **held_out**: `{"unsupported_component": 5}`
- **ood**: `{"unsupported_component": 4}`
- **rico**: `{"needs_direction_change": 35}`
- **smoke**: `{"unsupported_component": 3}`

## `twotower_prompt_ast`

- alphabet_fingerprint: `a429e7ddd2ec3b781fa7c9e7b70ca470437aa694d7d3833f76c3791a00ab149d`
- seed_id: `twotower_prompt_ast.full_mask_seed`

| suite | status | reachable_fraction | decided | unknown_budget |
| --- | --- | --- | --- | --- |
| adversarial | not_applicable | None | None | None |
| held_out | not_applicable | None | None | None |
| ood | not_applicable | None | None | None |
| rico | not_applicable | None | None | None |
| smoke | not_applicable | None | None | None |
| train | not_applicable | None | None | None |

> This variant has no discrete edit-action space to search: it denoises a fully-masked target directly via iterative MaskGIT-style denoising (models/twotower.py) rather than applying a bounded sequence of edits to a seed program. 'Bounded seed-to-target edit-path search' is not a meaningful measurement for this variant at all, independent of engineering effort -- this is a structural non-goal, not a deferred one.
