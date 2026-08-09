# VCE-007 (SLM-459): metamorphic prompt/program and positive-equivalence generators

New module: [`src/slm_training/data/semantic_contrast/metamorphic.py`](../../src/slm_training/data/semantic_contrast/metamorphic.py).
Companion to [`semantic-contrast-corpus-v1.md`](semantic-contrast-corpus-v1.md) /
[`semantic-contrast-corpus-v2-notes.md`](semantic-contrast-corpus-v2-notes.md)
(VCE-006/SLM-448), which this extends rather than duplicates. Deliverable is
five deterministic generator functions plus regression tests that empirically
prove each declared invariant on real generated programs -- not yet a
persisted corpus/CLI (no `rejected.jsonl`/`summary.json` writer exists for
this module; see Follow-up).

## The five families and what each reuses

| Family | Function | Reuses | Declares |
| --- | --- | --- | --- |
| Alpha-rename | `generate_alpha_rename_case` | Real `canonicalize_plan` renaming logic (`data/semantic_plan/canonicalize.py`) | `expected_changed=()`; every declared factor + compiled surface + verdict invariant |
| Reorder declarations | `generate_reorder_case` | `positive_control_sibling_reorder` (VCE-006, `semantic_contrast/transforms.py`) | `expected_changed=(surface, topology)`; verdict invariant |
| Prompt single-fact edit | `generate_prompt_single_fact_edit_case` | `extract_prompt_requirements` (SGS-004) | exactly the facts naming the edited component change |
| Prompt paraphrase invariance | `generate_prompt_paraphrase_case` | `extract_prompt_requirements` (SGS-004) | fact-set fingerprint invariant |
| AST rewrite equivalence | `generate_ast_rewrite_equivalence_case` | `dsl.analysis.optimize.optimize` (self-certifying `semantic_fingerprint` equality) | surface changes; `semantic_fingerprint` + verdict invariant |

Every case also carries `root_family_id`/`split` from `root_family_for` +
`RootFamilySplitPolicyV1` -- the identical identity scheme
`harnesses.train_data.semantic_counterfactuals.root_family_for` (SLM-366/
DSH2-06) already established, reused verbatim rather than inventing a second,
incompatible lineage scheme, per this repo's reuse-first convention.

## Honest findings during implementation

- **Alpha-rename only exercises real canonicalization on non-gold plans.**
  `SemanticPlanV1.compile_to_baseline()` is `True` for every gold-provenance
  plan (all plans `OpenUISemanticPlanExtractor` produces), which makes
  `canonicalize_plan` a no-op -- the real `_role_renames`/`_symbol_renames`
  logic never runs. `generate_alpha_rename_case` returns `None` on a gold
  plan rather than fabricate a case that can't prove what it claims;
  `test_alpha_rename_rejects_gold_provenance_plans` asserts this. The case
  itself is built against a plan with `identity.provenance="predicted"`
  (mirroring the existing non-gold test fixture pattern in
  `tests/test_data/test_semantic_plan_extraction/test_canonicalize.py`).
- **role_id/symbol_id are genuinely invisible to the compiled surface.**
  Verified empirically, not assumed: `PlanSeedBuilder` keys internal dicts by
  these ids but only ever *renders* `component_family` and bound content
  (`:slot_N`, itself derived from declaration *position*, not the symbol_id
  string) into the OpenUI text. Renaming every role_id/symbol_id with fresh,
  position-derived spellings produces a byte-identical compiled seed and an
  identical `meaningful_verdict` across every sampled source in the test
  fixture (6/6 sources; see `test_alpha_rename_leaves_surface_and_fingerprints_unchanged`).
- **`flatten_single_child` is deliberately excluded.** `optimize()` supports
  a flatten rewrite that changes `semantic_fingerprint` by design (per that
  module's own certification distinction) -- a UI-nesting change, not a pure
  equivalence. This generator only uses `elide_defaults`/`drop_dead_bindings`
  via `OptimizeOptions(flatten_single_child=False)`, directly satisfying "UI
  order/accessibility-sensitive rewrites are not assumed safe."
- **Paraphrase invariance is honestly scoped to the extractor's actual
  guarantee.** `extract_prompt_requirements` (SGS-004) is a deterministic,
  rule-based extractor, not an NLU system. It is provably invariant to
  whitespace/declaration order (verified in
  `test_prompt_paraphrase_invariant_to_whitespace_and_order`) -- the
  generator's docstring says explicitly not to claim general natural-language
  paraphrase understanding from this.

## Evidence

`pytest tests/test_data/test_semantic_contrast_metamorphic.py` -- 9/9 passing,
each assertion checked against real `ProgramGenerator`-produced sources (not
hand-authored fixtures alone), reusing the same generator config as the
existing `semantic_contrast` test suite. Combined with the untouched 7 VCE-006
tests and the runtime-shard test: 21/21 passing across
`tests/test_data/test_semantic_contrast.py`,
`tests/test_data/test_semantic_contrast_metamorphic.py`, and
`tests/test_scripts/test_verify_semantic_contrast_runtime.py`.

## Follow-up (not attempted here)

- No persisted-corpus CLI/writer for metamorphic cases exists yet (unlike
  `SemanticContrastBuilder`, which writes `pairs.jsonl`/`summary.json`/etc.).
  If VCE-008 (leakage/dedup/topology/OOD split audits, which this issue
  blocks) needs a durable metamorphic dataset rather than on-the-fly
  generator calls, that's a follow-up builder/CLI, not a generator-function
  gap.
- The ownership map (`repository-ownership-map.md`) also names
  `counterfactual_replay` (`harnesses/preference/counterfactuals.py`'s
  judge-based replay verifier) as an extension point for VCE-007. This
  implementation does not touch that file; wiring these generators into the
  judge-based replay path is separate follow-up work.
