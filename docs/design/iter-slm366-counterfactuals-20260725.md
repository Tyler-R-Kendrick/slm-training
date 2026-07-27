# DSH2-06 one-fact semantic counterfactuals + hard grounding contrasts (SLM-366)

**Decision:** supported at the contract-fixture evidence level.
`harnesses/train_data/semantic_counterfactuals.py` generates hard-valid
prompt/target pairs that isolate exactly one declared semantic fact — role,
order, cardinality, closed value, required child, or relation — rather than
prompt style or component inventory. For one verified `SemanticFrameV1`
(DSH2-02, SLM-363) it mutates one fact on the canonical AST, compiles a new
target satisfying the *changed* frame, renders matched-style length-bounded
prompts for both sides through the DSH2-03/DSH2-04 provider contract
(SLM-364/SLM-365 offline fixture provider — no network, no real LLM), and
admits the pair only when the one-fact proof and every surface control pass.

Machine-readable evidence:
[`iter-slm366-counterfactuals-20260725.json`](iter-slm366-counterfactuals-20260725.json).

## One fact at a time, never guessed

`FACT_DIMENSIONS = (role, order, cardinality, closed_value, required_child,
relation)`. `mutate_program` applies the typed mutation for one dimension on
the canonical AST and re-emits DSL source through a minimal declared emitter
(any node shape outside the DSH2-02 declared set fails closed). Each mutation
uses only schema/grammar-licensed alternatives:

- **closed_value** — flip a bool literal (or a schema-declared closed-domain
  scalar) to its declared complement;
- **order** — swap the first adjacent element siblings of different component
  types in one list slot;
- **cardinality** — duplicate the last child of one element-list slot
  (count N → N+1);
- **role** — exchange the first children between two element-list slots of
  one parent (per-slot cardinalities unchanged);
- **required_child** — replace a singleton-slot child with a declared
  alternative (`SINGLETON_CHILD_ALTERNATIVES`, Input → TextArea); anything
  undeclared rejects `no_mutation_site` rather than guessing;
- **relation** — retarget a state read to a different *declared* state.

## One-fact proof

`one_fact_proof` compares the two derived frames through **path-insensitive
per-dimension projections** (`dimension_projection`): role/order/cardinality/
required_child over typed parent→child edge sets, closed_value as a
(prop, value) set over required closed-domain facts, relation as
(kind, target) plus effects. A pair is admitted only when:

1. the declared dimension's projection diff is non-empty and **every other
   dimension's diff is empty** (two differing dimensions →
   `multi_dimension_diff`, tested);
2. both targets parse + validate + canonicalize, and their canonical
   fingerprints differ (`identical_canonical_targets` otherwise);
3. the inverse mutation round-trips the counterfactual AST back to the source
   canonical AST (`inverse_roundtrip_failed` otherwise).

## Matched-style prompts + surface controls

Both prompts are rendered in the same style (default `concise`) through the
SLM-365 provider contract and grounding floor, so style and component
inventory are controlled by construction. `check_pair_prompt_controls` then
enforces, cross-checked against **both** frames' declared leak detectors:

- **length bound** — relative prompt-length gap ≤ 0.4
  (`length_out_of_tolerance`);
- **marker surfaces** — placeholder markers never render (leak floor); state
  reads are neutralized to declaration ordinals ("declared state 1"), so a
  retargeted read still reads differently while `$ident` surfaces never
  appear (`marker_surface_cue`); permuting every placeholder in both sources
  (`marker_permutation` / `permute_marker_surfaces`) leaves prompts and the
  proof byte-identical — surfaces are provably not a cue (tested);
- **DSL / digest / provenance leaks** — `leak_detected`, including
  cross-contamination that passes one frame's own floor but names a component
  only the other target reveals (tested);
- **style mismatch** — `style_mismatch`.

**Stop rule.** Any uncontrolled semantic or surface cue rejects with a code
from `REJECTION_CODES` and is never counted as grounding evidence.

## Fixture results (all six dimensions admitted)

| Dimension | Mutation | Dimension diffs (declared / all others) | Length ratio |
| --- | --- | --- | --- |
| role | Form buttons↔fields first children exchanged | role 4 / 0 | 0.0 |
| order | Stack children [text, separator] swapped | order 2 / 0 | 0.0 |
| cardinality | last Stack child duplicated (2 → 3) | cardinality 2 / 0 | 0.195 |
| closed_value | Separator `decorative` true → false | closed_value 2 / 0 | 0.003 |
| required_child | FormControl input Input → TextArea | required_child 2 / 0 | 0.005 |
| relation | Callout visible read $s0 → $s1 | relation 2 / 0 | 0.0 |

All pairs: both targets compiler-valid, canonical fingerprints differ, inverse
round-trip passes, split `train` under one root family.

## Hard negatives + baseline

- **Embedding-close/action-distant** (`embedding_close_negative`): the
  order-swap pair's prompts have token-multiset Jaccard **1.0** while their
  correct targets differ structurally (order projection diff 2) — surface
  similarity is not a target cue.
- **Inventory-matched topology** (`topology_negative`): `Stack[Card[Text],
  Separator]` vs `Card[Stack[Text], Separator]` — identical component
  multiset, different parent→child topology; inventory mismatches and
  same-topology candidates reject.
- **Constant/frequency-only baseline** (`constant_baseline_accuracy`):
  exactly **0.5** (chance) by construction — each pair contributes one
  source-labeled and one counterfactual-labeled side.

## Split family

Source and counterfactual derivatives share one root split family
(`root_family_for` over the source canonical target);
`RootFamilySplitPolicyV1` assigns both to the same split and
`pair_artifact_nodes` links the counterfactual to its source parent. The
artifact graph quarantines cross-split surface reuse (`cross_split_exact`),
and a drifted family identity is rejected by the split policy itself
(tested).

Tests: `tests/test_harnesses/train_data/test_semantic_counterfactuals.py`
(44 tests: per-dimension admission, compiler validity + distinct fingerprints,
one-fact proof incl. two-dimension rejection, matched-style/length control,
both negative kinds, marker-permutation invariance, coded stop-rule
rejections, split-family + cross-family rejection, baseline at chance,
determinism, projection path-insensitivity).
