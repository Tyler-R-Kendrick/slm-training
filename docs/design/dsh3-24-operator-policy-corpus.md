# DSH3-24 operator-policy rows from collapse hard negatives

SLM-399 (DSH3-24) is milestone M6's second decision for "DSH3 — CAP2
Compiler-Owned AST Operators": can the existing replay-authoritative
collapse artifacts (`slm_training.dsl.operators.collapse`) become leak-free
supervision with real order-sensitive negatives, instead of a new synthetic
corpus? It builds directly on the SLM-397 sanitized `OperatorPolicyInputV1`
boundary (`src/slm_training/models/operator_policy_view.py`) and the
existing `CollapsedInstructionV1` / `CollapsedHardNegativeV1` evidence
`collapse_conversation_trace` already computes. It adds no new hard-negative
mining and no synthetic corpus generator — it only re-projects evidence that
already exists.

## Contract

`src/slm_training/data/flow/operator_policy_corpus.py`:

* `OperatorPolicyRowV1` — one leak-free supervision row per collapsed step:
  state/reference-table/legal-set identity (all content-addressed digests),
  the SLM-397 sanitized `policy_input` (`OperatorPolicyInputV1.to_dict()`,
  digest-pinned via `policy_input_digest` and recursively forbidden-field
  checked at construction — not just when it was first built), the accepted
  `accepted_operator_id` / `accepted_action_row` / `accepted_argument_rows`
  (row-local joins into `policy_input`, exactly like SLM-397's own joins),
  `accepted_application_id` (the recorded evidence's own digest, kept for
  provenance — never itself a model-input field), and an optional
  `OperatorPolicyHardNegativeV1`.
* `OperatorPolicyHardNegativeV1` — one collapse-derived adjacent-swap
  negative, re-projected onto *this* step's freshly re-enumerated action
  rows: `outcome` (`HardNegativeOutcome.CONFLICT` or `DIFFERENT_RESULT`,
  unchanged from `collapse.py`), `alternate_operator_id` (the swapped-in
  operator), `alternate_action_row` (its row in *this* step's live action
  rows, or `None` if it isn't currently live), and the original
  `conflict_code` / `observed_final_state_digest` evidence.
* `build_operator_policy_rows(trace, collapse, authority_resolver, split)` —
  walks each collapsed step and, for each one:
  1. Reads the actual trace state/reference-table at that step
     (`trace.node(trace.turns[step_index].input_state_id)`) — never the
     collapse's own recorded evidence.
  2. Calls `enumerate_operator_legal_set` **fresh** against that live
     state/reference-table/registry. This is the "re-enumerate current live
     membership rather than copying planner membership" requirement: even
     though the outcome is the same registry and (for these fixtures) the
     same result, the membership is recomputed, not read off `collapse`.
     The caller supplies the same positive `max_combinations_per_operator`
     bound used to create the source trace; reprojection never silently falls
     back to an unbounded legal-set enumeration.
  3. Builds the SLM-397 sanitized view (`build_operator_policy_input`) over
     that fresh legal set.
  4. Locates the recorded application inside the fresh legal set by
     `application_id`. If it isn't there, the step is **rejected**
     (`RowRejectionKind.ACCEPTED_ACTION_NOT_LIVE`), never forced in.
  5. Remaps accepted action/argument and hard-negative rows through
     `OperatorPolicyInputV1.canonical_row_maps()` before storing the canonical
     `policy_input` payload, so every evaluator-only row label still joins to
     the persisted candidate it names.
  6. Looks up whether this step is the *left* side of one of
     `collapse.hard_negatives`' `swapped_step_indices` pairs, and if so
     re-projects it onto this step's fresh action rows.
* `OperatorPolicyCorpusQualityReportV1` / `build_operator_policy_corpus` —
  aggregates accepted/rejected counts, rejection-reason counts, hard-negative
  outcome counts, and the positive-only-row count (rows with no certified
  negative) across one or more collapses' rows.

## `CONFLICT` stays distinct from `DIFFERENT_RESULT`

Both outcomes come straight from `collapse.py`'s own `HardNegativeOutcome`
enum, unchanged. `CONFLICT` means the swapped order didn't even execute
(unsafe/non-executable — `conflict_code` is set, no final-state evidence
exists). `DIFFERENT_RESULT` means the swapped order executed but landed on a
different final state (executable but target-inconsistent/order-sensitive —
`observed_final_state_digest` is set, no conflict code). `OperatorPolicyHardNegativeV1.__post_init__` enforces the same
exactly-one-of-two-evidence-shapes invariant `CollapsedHardNegativeV1`
already enforces.

## A registry-identity finding from building the adversarial test

While building the rejection test (`test_accepted_action_absent_from_a_re_
enumerated_library_is_rejected`), re-enumerating the *same* operator
(`openui.set_x`, unchanged declaration and unchanged `execute`) under a
**different** `OperatorLibraryV1` (one missing `openui.set_y` entirely)
produced a **different** `application_id` for the *unaffected* operator too
— not just for the missing one. The application's `proof.
compiler_result_digest` is sensitive to the registry's overall composition
(`registry_fingerprint`), not just to the one operator being dry-run. This
is consistent with — and further motivates — `OperatorLegalSetV1.
registry_fingerprint` and `build_operator_policy_input`'s own
`policy_view.registry_mismatch` fail-closed check (SLM-397): comparing
application IDs across two different registries is not meaningful even for
an operator that looks unchanged, so the rejection test instead keeps the
registry identical and makes `openui.set_y`'s *execute* unconditionally
fail — the realistic "same system, behavior changed between recording and
re-enumeration time" case.

## Verification matrix

Covered in `tests/test_data/flow/test_operator_policy_corpus.py`:

* `DIFFERENT_RESULT` — two unconditional-overwrite operators
  (`openui.set_x` → `:set.x`, `openui.set_y` → `:set.y`) whose swapped order
  both fully executes but disagrees on the final state; the negative is
  re-projected onto step 0 with a resolvable `alternate_action_row`
  (`test_different_result_hard_negative_is_reprojected_onto_step_zero`).
* `CONFLICT` — `openui.set_b` requires `openui.set_a`'s effect; swapping the
  order makes `set_b` fail outright on the root state
  (`test_conflict_hard_negative_is_reprojected_onto_step_zero`).
* Forbidden-field re-validation of every row's `policy_input`, and
  `policy_input_digest` self-consistency
  (`test_rows_carry_only_allowlisted_policy_input_fields`).
* Reject-not-force when the recorded application isn't live in a fresh
  re-enumeration (`test_accepted_action_absent_from_a_re_enumerated_library_
  is_rejected`).
* `OperatorPolicyCorpusQualityReportV1` aggregation
  (`test_corpus_quality_report_aggregates_hard_negatives_and_rejections`).

Regression: `tests/test_data/flow/test_bridge_corpus.py`,
`tests/test_models/test_legal_edit_batch.py`,
`tests/test_dsl/test_operator_conversation.py` (collapse tests live there;
see scope notes), `tests/test_dsl/test_operator_legal_set.py`,
`tests/test_dsl/test_operator_references.py`,
`tests/test_models/test_operator_policy_view.py`,
`tests/test_models/test_operator_feature_encoder.py` — all pass unchanged.

## Stop-rule disposition

The ticket's stop rule fires "if too few states have certified nontrivial
negatives" — retain fixture status, don't claim the negative-supervision gap
is closed at scale. This change delivers exactly that: two small, hand-built
fixture traces producing one `CONFLICT` and one `DIFFERENT_RESULT` negative
each. No claim is made that real verified conversation traces contain enough
non-commuting adjacent operator pairs to close the "zero-negative gap" at
production scale — that requires running this builder over the actual
verified conversation-trace population, which is out of scope here.
`claim_class: wiring` throughout; no train/eval/benchmark/ship claim.

## Scope notes (deliberately deferred)

* **No file-level corpus writer/loader.** `bridge_corpus.py`'s
  `write_corpus` / `load_corpus` (content-addressed JSON files on disk,
  independent replay validation on load) are not mirrored here — this
  change delivers the row/report contract and in-memory builder only. A
  follow-up can add the on-disk layer once real trace volume justifies it,
  reusing this module's `to_dict()`/digests as the wire format.
* **No dataset card.** Required once a real (non-fixture) corpus is built
  from verified conversation traces; premature for two hand-built fixtures.
* **`tests/test_dsl/test_operator_collapse.py` does not exist** in this
  repository — collapse's own tests live in
  `tests/test_dsl/test_operator_conversation.py` instead. The ticket
  template's test list was written generically; this doc records the
  substitution rather than silently diverging from it.
* Only adjacent-swap negatives (exactly what `collapse.py` already computes)
  are re-projected — no new negative-mining strategy is introduced here.

These are compiler/data-contract unit fixtures. No train, eval, benchmark,
matrix, checkpoint, model-card, ship-gate, or model-quality claim is
produced.
