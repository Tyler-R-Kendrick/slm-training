# SLM-394 / DSH3-19 — effect-derived merge conflicts

Date: 2026-07-25 (re-verified at HEAD 2026-07-26)
Status: fixture/wiring evidence only; no model or ship claim

## Decision

Does the substring test for delete/remove operator names misclassify branch
conflicts, and can conflict typing be derived entirely from exact effects and
declarations? **Yes on both counts.**

## What the name heuristic misclassified

The pinned-baseline classifier (`merge.py::_conflict_kind` at `5cd5b8b`)
labeled any overlapping pair `DELETE_MODIFY` when either operator ID contained
`"remove"` or `"delete"`. Two concrete misclassifications, now pinned by
metamorphic regression tests:

1. **False positive — misleading name, non-removal effects.**
   `openui.delete_name_only` (topology delta only, no `consumed_nodes`) merged
   against `openui.relocate` (balanced consumed+produced node flow, i.e. a
   move, not a removal). Name heuristic: `DELETE_MODIFY`. Effect-derived:
   `CHILD_ORDER` — nothing is deleted.
2. **Missed conflict — removal semantics without a delete-like name.**
   `openui.rewrite` (explicit `consumed_nodes`, a real removal) merged against
   `openui.rename_again` (property delta on the same target). Name heuristic:
   `same_node_incompatible_edit`. Effect-derived: `DELETE_MODIFY` with the
   exact target fingerprint.

Proofs: `tests/test_dsl/test_operator_merge.py::test_effect_derived_delete_conflict_ignores_operator_names`
and `::test_effect_classifier_uses_lineage_and_refuses_unproven_removal`.

## Fix (landed on main via PR #965, commit 5e182534)

- `ActionEffectV1` gained explicit `consumed_nodes` / `produced_nodes`
  (`dsl.operators.contracts` v15); `REMOVE_NODE` records consumed flow
  (`dsl.operators.local` v4); move/reparent record balanced consumed+produced
  flow (`dsl.operators.topology` v2).
- Public pure `classify_merge_effects` (exported from
  `slm_training.dsl.operators`) types conflicts from declarations, exact
  effects, and base-reference lineage only — operator IDs are never inspected.
  A removal requires consumed-node flow not restored by produced-node flow;
  unproven removal returns a conservative `UNSUPPORTED_EFFECT` with the target
  fingerprint. `STALE_REF` keeps priority over overlap classification.
- `merge_conversation_branches` routes through the classifier after replaying
  both edits through pack authority (`library.replay`); replay or provenance
  mismatch is a typed refusal, never a silent merge.
- No production code branches on operator-name substrings. The historical
  heuristic survives only as `_historical_label`, the comparison baseline
  inside `scripts/audit_merge_conflicts.py`.

## Measured results (re-run at HEAD for this iter)

`MergeConflictAuditV1` — `docs/design/slm-394-merge-conflict-audit-20260725/report.json`:

- legal-set coverage: `complete`; replayed exact actions: `66`;
  unordered pairs: `2145`; unexplained old/new disagreements: `0`.

| Label | Historical name heuristic | Effect-derived |
| --- | ---: | ---: |
| `child_order` | 152 | 152 |
| `delete_modify` | 56 | 56 |
| `disjoint` | 1352 | 1352 |
| `role_cardinality` | 582 | 582 |
| `same_node_incompatible_edit` | 3 | 3 |

On the registered OpenUI local/topology corpus both labelings agree, so the
measured production impact on today's corpus is nil — the corpus contains no
misleadingly named or renamed-removal operators. The misclassification proof
lives in the metamorphic fixtures above; the audit pins the invariant that
effect-derived typing reproduces every registered label without name
semantics. Merge legality is unchanged (no previously refused pair
auto-merges; the stop rule was never triggered).

## Verification

- `pytest -q tests/test_dsl/test_operator_merge.py tests/test_dsl/test_operator_contracts.py tests/test_scripts/test_audit_merge_conflicts.py` → 25 passed
- Neighboring: `pytest -q tests/test_dsl/ -k "operator or merge or transaction or topology or local"` → 244 passed
- AgentV bundle (3/3 criteria pass): registered-pair-coverage,
  conservative-incomplete-effects, explained-disagreements — refreshed at HEAD
  in the audit directory above.

## Caveats

- Fixture/wiring evidence only; not a ship or quality claim.
- Zero registered-corpus disagreements means the classifier change is
  semantics-preserving on current operators; its value is removing
  name-dependent authority, proven on adversarial renamed/misnamed fixtures.
- `classify_merge_effects` stays fail-closed: non-EXACT coverage, empty refs,
  declaration/effect-signature mismatch, stale lineage, or unproven removal
  all yield typed conflicts, never auto-merges.

## Version stamp

```json
{
  "code_commit": "5b49224cb2f5fe7e1a5e5793e4de2868025274cf",
  "code_dirty": true,
  "components": {
    "dsl.operators.contracts": "v15",
    "dsl.operators.merge": "v3",
    "evals.agentv": "v5"
  },
  "stamp_schema": "version_stamp/v1",
  "stamped_at": "2026-07-26T01:20:48.955456+00:00"
}
```
