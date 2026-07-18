# LDI0-02 — DecisionEventV2 (state / action-evidence / objective-view)

Date: 2026-07-17
Status: **schema + materializers + V1 migration, with tests; no event mining,
no trainer change, no checkpoint, and no model-quality or ship claim.**

## What and why

`DecisionEventV2` (`src/slm_training/harnesses/preference/decision_events_v2.py`)
replaces V1's frozen `good_token_ids` / `bad_token_ids` payload as the *primary*
semantic corpus contract with a three-part model that separates concerns V1
conflated — the fix for the E284 blocker (*stable grammar-state support does not
imply objective/action-partition support*):

* **`DecisionStateV2`** — the exact, replayable model decision state plus immutable
  runtime identities. Its `state_id` is a canonical `content_sha` over state and
  identity **only** (never labels, rollout outcomes, or candidate order), and
  generalizes V1's TwoTower-only `canvas_ids`+`position` shape to `twotower` and
  `causal` (prefix) tracks.
* **`ActionOutcomeV2`** — append-only, per-candidate verifier evidence. The complete
  ordered **G0–G12** gate vector is preserved verbatim (validated, never
  set-collapsed). Merge deduplicates by content identity, not array order.
* **`ObjectiveView`** — a pure, versioned materialization of one state's action
  table. Five materializers: `pareto`, `thresholded`, `single_best_worst`,
  `set_valued` (all `trainable=True`), and `constraint_shadow`
  (`trainable=False` — a semantic trainer must refuse legality-only evidence).

`migrate_v1_event` performs a one-way, deterministic, idempotent V1→V2 migration
that marks partial evidence **incomplete** and never fabricates rollout/verifier
vectors. V1 corpora and their loaders are untouched.

## Fixture-corpus demonstration

The executable version of this walk-through is
`tests/test_harnesses/preference/test_decision_events_v2.py` (26 tests). One exact
state with two independently-observed actions:

1. **One exact state** — a `DecisionStateV2` for a `component` decision with
   `legal_action_ids = (4, 9, 10)`. Its `state_id` is fixed by the state/identity
   fields; constructing it twice yields the same id
   (`test_two_label_samples_of_one_state_share_one_state_id`), and it is unchanged
   by how action evidence is later ordered or augmented
   (`test_state_id_is_independent_of_action_evidence_order_and_augmentation`).
2. **Multiple independently replayed actions** — action `4` observed with reward
   `0.9`, action `9` with reward `0.1`, each carrying its own complete G0–G12
   vector; a re-observation of identical evidence collapses under the append-only
   merge, new evidence appends
   (`test_append_only_merge_dedups_by_content_not_order`).
3. **Two materializers, different views from the same evidence** — `pareto` labels
   `good={4}`, `bad={9}`, `unobserved={10}` and is trainable; `constraint_shadow`
   labels `good={}`, `bad={}`, `ambiguous={4,9}` and is **not** trainable
   (`test_pareto_view_is_trainable_and_partitions_actions`,
   `test_constraint_shadow_view_is_not_trainable`).
4. **Stable state fingerprint, changing objective fingerprint** — the manifest
   (`decision_state_manifest`) fingerprints states, action evidence, and objective
   views separately and order-independently
   (`test_manifest_fingerprints_are_separate_and_order_independent`): re-ordering
   the evidence rows leaves the outcome fingerprint unchanged, while the state,
   outcome, and view fingerprints differ from one another.

## Validation (fail-closed)

Enforced and tested: unknown record fields rejected; a tampered `state_id`
rejected; `split` must derive from `group_id`; a TwoTower state requires
`canvas_ids`, a causal state requires `context_ids`; every present verifier vector
must carry the complete ordered G0–G12 gate set with valid statuses; objective-view
partitions must be disjoint; a materializer rejects an outcome whose `state_id`
does not match the state; atomic, duplicate-safe JSONL writes.

## Honest scope / caveats

- **No model or ship claim.** This is a corpus/record schema plus pure
  materializers and a migration. No event is mined, no trainer consumes V2 yet, no
  checkpoint or eval is produced.
- **Named independent-judge rationale** beyond the G11 gate status, and a
  full model-driven **logit-replay** integration fixture (versus the schema-level
  causal/TwoTower state construction covered here), are the honest follow-ons; the
  state records already carry the prefix/canvas + policy/tokenizer/decode/verifier
  identities required to replay.
- Verified with `pytest tests/test_harnesses/preference/test_decision_events_v2.py`
  (20) and `tests/test_harnesses/preference/test_local_decisions.py` (V1 unchanged,
  11); `ruff` and `python -m scripts.repo_policy` clean.
