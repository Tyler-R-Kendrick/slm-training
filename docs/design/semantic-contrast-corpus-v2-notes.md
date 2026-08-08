# VCE-006 (SLM-448): semantic-contrast corpus factor-family extension

Companion to [`semantic-contrast-corpus-v1.md`](semantic-contrast-corpus-v1.md).
This is a **builder/harness code change**, not a republish of the immutable
[`openui_hard_valid_v1`](../../src/slm_training/resources/data/eval/openui_hard_valid_v1/)
corpus, which is untouched (`manifest.json["immutable"] = true`; `DataStore.publish`
fails closed on a second publish to the same `dataset_id`). Every train/eval/
bench/matrix run in this repo obeys `MAX_RUN_MINUTES = 3`
(`src/slm_training/levers.py`), so a full 210-source, 29-shard, Chromium
runtime-verified republish is out of scope for a single run and is left as
follow-up data-build work (tracked below). What shipped here is real,
runnable builder code plus **fixture-scale evidence** that the new factor
coverage actually admits — not a ship claim for a new corpus version.

## What changed

1. **`topology_reparent` bug fix.** The transform picked `edges[0]` /
   `edges[1]` unconditionally. For the generator's actual output shape (root
   -> one wrapper layer -> leaves), those two edges always share a
   child/parent boundary (edge 0's child *is* edge 1's parent), so
   `child == new_parent` on every real source and the transform silently
   returned `None` every time -- this is the root cause of the "topology
   n_total mostly 0" gap `semantic-contrast-corpus-v1.md`'s honest caveats
   already called out. The fix searches every `(child edge, candidate
   container parent)` combination, restricts new-parent candidates to role
   slots whose component family is a known container (mirroring
   `OpenUISemanticPlanExtractor._CONTAINER_TYPES`), and skips any move that
   would leave the old parent with zero children (`PlanSeedBuilder` renders a
   childless container without its required `children` prop at all, which is
   schema-invalid -- see the transform's docstring).
2. **New positive-equivalence control: `positive_control_sibling_reorder`.**
   Swaps the declared order of two children under the same parent. Every
   plan this builder extracts is gold-provenance, so
   `SemanticPlanV1.compile_to_baseline()` is true and `canonicalize_plan`
   returns the plan unchanged instead of order-normalizing it -- sibling
   order is therefore a real, single-factor (`topology`) delta, and
   `PlanSeedBuilder` renders children in raw edge-list order, so the
   compiled surface text differs too. Both sides must still pass
   `binding_aware_meaningful_v2`, directly answering the acceptance
   criterion "add positive equivalence controls so the evaluator is not
   trained to reject every structural change."
3. **Prompt-contract facts (SGS-003/004) attached to every record.** Each
   `SemanticContrastRecord.meta` now carries `prompt_requirements`
   (`PromptSemanticRequirementsV1.to_dict()`, extracted gold-blind from the
   shared prompt via `extract_prompt_requirements`) and
   `prompt_requirements_fingerprints` (`requirement_fact_fingerprints`).
   These facts are `authority="advisory-learned"` by construction and are
   never used to gate admission -- attached as provenance only.
4. `BUILDER_VERSION` `2.0.1` -> `2.1.0` (additive `meta` keys, new taxonomy
   entries, changed `topology_reparent` admission behavior).

## Fixture-scale evidence

Recipe (deterministic, ~15s):

```bash
python -m scripts.build_semantic_contrasts --dataset-id openui_hard_valid_v2_fixture \
  --source-count 30 --seed 3 --wide-source-grid --strict-delta
```

Full stamped summary:
[`semantic-contrast-corpus-v2-fixture-evidence.json`](semantic-contrast-corpus-v2-fixture-evidence.json).

| transform_id | admitted | notes |
| --- | --- | --- |
| `positive_control_sibling_reorder` | **18** | new; both sides pass; verified via `tests/test_data/test_semantic_contrast.py::test_sibling_reorder_admits_as_a_positive_equivalence_control` |
| `topology_reparent` | 0 (7 attempted, was 0 attempted before this fix) | compiles cleanly now; rejected via `negative_passed` -- see honest gap below |
| `topology_delete_leaf` | 0 (24 attempted) | unchanged from v1; `compilation or verifier rejection` |
| `binding_swap_symbol` / `_reverse` | 18 / 16 | unchanged behavior |
| `content_swap_family` | 14 | unchanged behavior |
| `positive_control_identity` | 30 | unchanged behavior |

Regression coverage (`tests/test_data/test_semantic_contrast.py`, 11 tests,
all passing): the 7 pre-existing tests plus
`test_topology_reparent_moves_a_role_into_a_sibling_container`,
`test_topology_reparent_refuses_to_empty_a_container`,
`test_sibling_reorder_admits_as_a_positive_equivalence_control`,
`test_prompt_requirements_are_attached_to_every_record`.

## Honest gaps (not closed by this change)

- **`topology_reparent` still admits zero negatives.** It now compiles (real
  progress -- previously it never even produced a valid candidate on a real
  source), but `binding_aware_meaningful_v2` has no nesting-depth/parent
  signal: it checks component inventory, placeholder bindings, and prompt
  contract, not tree shape. Moving a leaf between two structurally fungible
  `Stack` containers doesn't change inventory or bindings, so the evaluator
  correctly (by its own current contract) still passes it. Closing this
  needs either a topology-aware evaluator check or a container-choice signal
  in the prompt contract -- flagged as follow-up, not attempted here to
  avoid touching `evals/meaningful_program.py` (used far beyond this corpus)
  under a single-issue PR.
- **`topology_delete_leaf` is unchanged** (still 0 admitted in this sample;
  matches the `openui_hard_valid_v1` production corpus's own
  `rejected.jsonl`, which shows the same `compilation or verifier rejection`
  pattern at full scale).
- **Form/state/reference source coverage was not attempted.** Widening the
  generator's `components` to include `Form`/`Input` (or bare `Stack`, which
  a probe during this change confirmed the *default* greedy `_choose()`
  candidate renders as a schema-valid but empty `Stack([])`, requiring
  `generate_uniform`/larger sample counts to reach non-degenerate nesting)
  is real, separate generator-surface work with its own risk profile; left
  as follow-up rather than rushed into this PR.
- No new immutable dataset was published; `openui_hard_valid_v1` numbers in
  `semantic-contrast-corpus-v1.md` are unchanged and still authoritative for
  the shipped corpus.

## Follow-up

- Give `binding_aware_meaningful_v2` (or a sibling check) a topology/nesting
  signal so `topology_reparent` negatives can actually be admitted.
- Widen source generation to include `Form`/`Input` and genuine multi-level
  containers for form/state/reference factor coverage.
- If/when the above lands, a real `openui_hard_valid_v2` corpus can be
  published (new immutable `dataset_id`, full source count, runtime shards)
  following the `openui_hard_valid_v1` recipe in `semantic-contrast-corpus-v1.md`.
