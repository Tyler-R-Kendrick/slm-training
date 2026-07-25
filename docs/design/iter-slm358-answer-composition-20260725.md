# DSH1-06 typed answer composition with scope-safe AST combinators (SLM-358)

**Decision:** supported at the deterministic contract-fixture level. Verified
answers compose into larger objects, lists, scopes, and documents through
typed AST operations only — never raw string concatenation — with
deterministic alpha-renaming, fail-closed admission, and content-addressed
lineage.

Machine-readable evidence:
[`iter-slm358-answer-composition-20260725.json`](iter-slm358-answer-composition-20260725.json).

## Combinator registry

`answer_composition_registry/v1`
(`src/slm_training/harnesses/train_data/answer_composition.py`) registers five
generation-only `CombinatorV1` entries — `APPEND_STATEMENT`,
`INSERT_CHILD`, `MAKE_LIST`, `WRAP_NODE`, `COMPOSE_DOCUMENT` — each with a
typed domain/codomain and explicit pre/postconditions. Combinators take
parsed statement-AST objects (`production_codec` bindings: `element` /
`ref` / `list` / literal nodes) and return a `CompositionResultV1`
(`ok`, `composite_source`, `source_answer_ids`, `combinator_version`,
before/after AST digests, and a proof or a rejection `{code, reason}`).
Answer source text is only ever *parsed*; the composite surface is produced
by a deterministic AST renderer, the same discipline as
`decode_productions`. An `ast`-module audit test scans the composition
module itself and rejects any `+` / f-string / `str.join` over program text.

## Scope-safe binder resolution

Binder collisions are resolved by deterministic alpha-renaming following the
marker-table authority (`marker_tables` opaque-namespace scheme): the full
rename map is computed up front from content digests and applied to every
definition and reference in one pass, so no partially-renamed intermediate
is observable. The map rides in the proof report; the collision test
(composing two answers that both bind `root` and `v0`) shows the renamed
root referencing the renamed binder with zero stale references, and
byte-identical results across repeated runs.

## Admission and stop rules

Every application re-parses the composite through the official parser
(`dsl.parser.validate`) and canonicalizes it (`dsl.canonicalize`), requiring
a canonical fixed point; failures reject with stable `CompositionCode`s
(`unresolved_ref`, `cardinality_violation`, `invalid_target`,
`incompatible_roots`, `cross_split_source`, `illegal_intermediate`,
`composite_parse_failure`, `composite_validation_failure`,
`canonical_roundtrip_mismatch`, `precondition_violation`). Stop rules are
exercised: inserting a child into a leaf component and wrapping a binder
that other statements still reference (which would silently change the
parents' meaning) are both rejected as `illegal_intermediate`; typed v0.5
roots and cross-split sources fail closed with `incompatible_roots` /
`cross_split_source` (`RootFamilySplitPolicyV1`).

## Lineage

Admitted compositions persist as content-addressed `composite_answer` nodes
in the artifact graph (`persist_lineage`): parent ids are the source answer
ids, the payload carries combinator version + before/after AST digests +
proof, and the artifact id is deterministic (idempotent rewrite verified).
Rejected compositions have no lineage and `persist_lineage` refuses them.

## Results

- `tests/test_harnesses/train_data/test_answer_composition.py`: 15 passed
  (all five combinators round-trip canonically; deterministic atomic
  alpha-rename on collision; no-string-concatenation audit; incompatible
  roots / cross-split fail closed; illegal-intermediate stop rule ×2;
  pre/postcondition violations ×5; lineage content-addressed +
  deterministic; rejected compositions persist no lineage).
- Train-data + canonicalizer suites: see commands in the JSON evidence;
  this change adds no new failures over the branch baseline.
- `python -m scripts.verify_version_stamps --check`, `repo_policy`, ruff,
  `git diff --check`: passed.

Registry: new component `harness.experiments.slm358_answer_composition` v1;
`harness.train_data` bumped to v22 (new module under its watched directory).

Claim limits: fixture-scale contract evidence only — no corpus publication,
no model evaluation, no ship-gate claim. Composition admits only through the
official parser + canonicalizer; anything else is a rejection record, never
an accepted output.
