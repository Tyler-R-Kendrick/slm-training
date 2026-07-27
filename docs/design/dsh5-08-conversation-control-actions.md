# DSH5-08 conversation control-plane action set

SLM-416 (DSH5-08) is milestone M3's decision for "DSH5 — Bulk Operators,
Transactions & Control Plane": how should STOP, UNDO, REDO, CHECKOUT, FORK,
and MERGE_BRANCHES be represented and selected without conflating
conversation navigation with AST mutation?
`src/slm_training/dsl/operators/control_actions.py` answers this by
defining `ConversationControlKind` as a disjoint candidate space from
`AstOperatorV1`, enumerating its legal members directly from the current
`ConversationTraceV1` branch graph (plus caller-proposed
`MergeCandidateV1` triples for `MERGE_BRANCHES`, resolved through the
existing `merge.py`/`sequence_merge.py` machinery), and executing every
legal action through the conversation/merge functions that already exist,
are already replay-proven, and are unchanged by this issue.

## Decision and hypothesis

A separate compiler-owned control-action legal set and policy head should
improve command correctness and stop/history calibration while preserving
AST operator semantics and replay authority. This issue builds the
compiler-owned half (legal set, execution, receipt) and a deterministic
baseline policy arm plus the report type a learned arm would be compared
against — it does not train or ship a learned control policy (see
"Scope notes" below).

## Why `ConversationOperation` isn't enough on its own

`conversation.py`'s own `ConversationOperation` enum
(`AST_EDIT`, `UNDO`, `REDO`, `CHECKOUT_STATE`, `FORK`, `COPY_STATE`,
`TRANSACTION_COMMIT`) already separates AST-mutating turns from
navigation turns structurally — that seam is real and this issue reuses
it rather than re-deriving it. What's missing is a **legal-set view**
over that seam: given the current trace, which of UNDO/REDO/CHECKOUT/FORK
are actually available right now, with which arguments, and how does
MERGE_BRANCHES (which isn't a `ConversationOperation` at all — a merge
produces a **fresh trace root**, not a turn on an existing trace, per
DSH3-09/DSH3-20) fit into the same enumerable, provable candidate space a
router or policy head could consume. `ConversationControlKind` and
`enumerate_conversation_control_legal_set` are that view.

## Legality — read directly off the trace, not re-derived

Each control kind's legality was established empirically from
`conversation.py`'s own precondition checks (every failure there raises
`ConversationTraceError` with a fixed message) and reproduced exactly,
never loosened or tightened:

| Kind | Legal iff | Source of truth |
|---|---|---|
| `STOP` | always | control-plane sentinel; never gated on trace shape |
| `UNDO` | `current.parent_state_id is not None` | `undo_conversation`'s own check (`"nothing to undo"`) |
| `REDO` | one entry per `node.parent_state_id == current.state_id` | `redo_conversation` requires `target.parent_state_id == current.state_id`; multiple legal targets arise naturally from multiple forks off one node |
| `CHECKOUT` | one entry per other trace node | `checkout_conversation_state` requires only `target != current` — least constrained of the four, deliberately |
| `FORK` | always, unconditionally, from any current state | confirmed empirically: `fork_conversation`'s only failure mode (`"fork branch already exists"`) depends on the *proposed* nonce, not the current state, so it cannot be pre-checked by a legal-set enumerator that isn't proposing a nonce |
| `MERGE_BRANCHES` | caller-proposed `(base, left, right)` triple resolves, branches differ, and the segment actually merges | not a `ConversationOperation`; legality is proven by literally invoking `merge_conversation_branches`/`merge_branch_application_sequences` |

Unlike `OperatorLegalSetV1`, this legal set never reports `PARTIAL`
coverage or truncates a bounded scan: every one of these candidate spaces
is already exactly enumerable from the trace in bounded time (there is no
combinatorial argument-slot product to bound, unlike an `AstOperatorV1`'s
argument domains). The one place `UNKNOWN` legitimately appears is
`MERGE_BRANCHES` with no `authority_resolver` supplied — the same
externally-injected, non-model dependency `merge_conversation_branches`
itself requires — proven by
`test_merge_branches_without_authority_resolver_is_unknown_not_unsupported`.

## MERGE_BRANCHES: gathering typed arguments from a trace, never raw content

`MERGE_BRANCHES`'s three arguments are **state IDs already present in the
trace**, never raw ASTs or a model-generated pair — `MergeCandidateV1`
holds exactly `base_state_id`/`left_state_id`/`right_state_id`. Turning a
candidate into a real merge attempt:

1. Resolve all three IDs via `trace.node(...)` (unknown ID → illegal,
   `"unknown_state_id"`).
2. Require `left.branch_digest != right.branch_digest` (`"same_branch"`
   otherwise).
3. **`_walk_branch_segment`** collapses each branch's `AST_EDIT` turns from
   `base` to its tip into an ordered `tuple[BranchEditV1, ...]`, exactly
   mirroring how `sequence_merge.py`'s own module docstring says a caller
   must collapse a real trace: skip the one `FORK` turn that establishes
   the branch (transparently, since it preserves state and is not
   itself evidence a merge needs), and return `None` — unsupported — for
   any other turn kind encountered (`TRANSACTION_COMMIT`, `COPY_STATE`, or
   a history-navigation turn), since none of those are representable by
   `BranchApplicationSequenceV1` today (an explicit DSH5-07 scope
   boundary, inherited here rather than worked around).
4. The collapsed segments feed **`merge_conversation_branches`** (exactly
   one step per side) or **`merge_branch_application_sequences`** (two or
   more steps per side) unchanged — this module adds no new merge logic,
   only the walk that turns trace evidence into their existing input
   shapes.
5. `authority_resolver`/`reference_table_builder` are supplied by the
   caller exactly as `merge_conversation_branches` already requires them —
   never inferred, never model-generated. This is the literal reading of
   "merge authority, not model-generated IDs" from the issue text.

A candidate is legal iff the resulting `BranchMergeDecisionV1`/
`BranchSequenceMergeDecisionV1.succeeded` — i.e. legality *is* running the
real, already-conservative conflict classifier, not a separate cheaper
proxy for it. `test_disjoint_merge_branches_candidate_is_legal_and_executes`
and `test_overlapping_merge_branches_candidate_is_illegal_with_typed_reason`
prove both directions, the latter surfacing the exact
`MergeConflictKind` (`same_node_incompatible_edit`) as the rejection
reason (`f"merge_conflict:{kind.value}"`) rather than a generic failure.

## Execution: a thin, typed dispatch — no new mutation logic

`apply_conversation_control_action` is a dispatch table over
`ConversationControlKind`, each branch calling exactly one existing,
already-tested function:

* `STOP` → no trace call at all; returns a receipt carrying only a typed
  `stop_reason` string (caller-supplied, e.g. `"budget_exhausted"`,
  defaulting to `"policy_decided"`) — mirroring `flow/termination.py`'s
  own discipline that every STOP exit path carries a distinct reason code,
  never a bare boolean (`sample_with_termination`'s `stop_reason`
  assignment per branch). `test_stop_result_carries_only_a_typed_reason_and_never_mutates`
  proves STOP never touches the input trace.
* `UNDO`/`REDO`/`CHECKOUT`/`FORK` → `undo_conversation`/`redo_conversation`/
  `checkout_conversation_state`/`fork_conversation`, unchanged.
* `MERGE_BRANCHES` → the same walk-then-merge path used for legality
  checking, so "legal" and "executed" always agree (no separate,
  potentially-diverging execution code path).

`ConversationControlResultV1` is the receipt: `__post_init__` enforces
that a `STOP` result carries only `stop_reason`, a `MERGE_BRANCHES` result
carries only `merge_decision`, and every other kind carries only
`output_trace` — this is a structural guarantee (raises `ValueError`
otherwise), not a convention. "Replay" for every kind reduces to replaying
the underlying primitive this module dispatched to
(`replay_conversation_trace` for the four history operations,
`replay_branch_merge`/`replay_branch_sequence_merge` for
`MERGE_BRANCHES`) — this issue adds no new replay machinery, since none
of these outputs need one beyond what already exists and is already
tested.

## Deterministic baseline arm and report

Per the acceptance criteria ("Learned control is adopted only on causal
held-out benefit; deterministic/API behavior remains available"),
`deterministic_control_priority` is a real, fully deterministic policy:
COMPLETE singleton bypass (mirrors the same `_check_forced` pattern used
by `models/local_action_head.py`/`legal_action_scorer.py` for AST operator
heads — if exactly one legal action exists across every kind, return it
forced, confidence 1.0) and abstain on an empty legal set, else a fixed
priority order (`MERGE_BRANCHES > REDO > UNDO > CHECKOUT > FORK > STOP`)
favoring information-committing actions over pure exploration or the
no-op. This ordering is a documented, inspectable default — not a claim
about optimal policy — and it is the "deterministic/API" arm the
acceptance criteria require to remain available regardless of what a
future learned arm does.

`ConversationControlPolicyReportV1` mirrors
`models.operator_termination.OperatorTerminationReportV1`'s shape exactly
(command/argument accuracy in place of that report's action-only accuracy,
plus `stop_brier`/`stop_ece`/`premature_stop_rate`/`late_stop_rate`). Its
calibration math (`_brier_score`/`_expected_calibration_error`) duplicates
`flow.termination.brier_score`/`expected_calibration_error`'s formulas
locally rather than importing them: that module also imports
`flow.reference.generator`, which depends on `numpy`, and every
`dsl/operators/` module — including this one — must stay importable
without `numpy`/`torch` for the numpy-free `python-static` CI lane.
`test_control_actions_module_does_not_import_flow_termination` guards the
regression. This keeps the whole `control_actions.py` module numpy/torch-free,
consistent with `dsl/operators/`'s existing convention.

## Coverage and what's excluded

**Covered, proven by `tests/test_dsl/test_conversation_control_actions.py`
(12 tests):**

* Root-trace legal set: only `STOP`/`FORK` legal; `UNDO`/`REDO`/`CHECKOUT`
  each report their typed rejection reason; `MERGE_BRANCHES` reports
  `UNKNOWN`/`"no_candidates_proposed"` with none proposed.
* Editable-history matrix: after one `AST_EDIT`, `UNDO` becomes legal and
  `CHECKOUT`'s single legal target is exactly the parent; after undoing,
  `REDO`'s single legal target is exactly the child just undone from —
  and executing that `REDO` action lands back on it.
* Fork creates a second, sibling `REDO` target visible from the shared
  parent — the "ambiguous redo" case the issue's matrix names explicitly.
* `MERGE_BRANCHES`: disjoint-edit candidate legal and executes to a merged
  state containing both branches' edits; same-target candidate illegal
  with the exact `MergeConflictKind`; missing `authority_resolver` yields
  `UNKNOWN`, not a false `UNSUPPORTED`.
* `STOP` never mutates and carries only its typed reason.
* `LegalControlActionV1` structurally rejects a mismatched target/merge-
  candidate shape for its kind (`REDO` without a target, `STOP` with one,
  `MERGE_BRANCHES` without a candidate).
* `deterministic_control_priority`'s forced/abstain bypass and fixed
  ordering, at two different trace positions.
* `ConversationControlPolicyReportV1.from_predictions` computes calibration
  correctly against a small synthetic prediction set, rejects mismatched
  input lengths instead of silently truncating via `zip`, and the module
  never re-acquires the numpy-pulling `flow.termination` import.

**Deliberately excluded (not a stop-rule trigger):**

* **A trained/learned control-policy head.** The issue's own stop rule
  ("If a learned unified control policy regresses command safety or adds
  no held-out benefit, retain deterministic/API control execution") frames
  training as *conditional on a benefit that must first be demonstrated* —
  this issue ships the compiler-owned legal set, the deterministic arm,
  and the report type a learned arm's output would be scored against
  (`ConversationControlPolicyReportV1.from_predictions` accepts any
  sequence of predicted/expected actions and stop probabilities,
  independent of how they were produced), so a future learned arm can be
  compared without further plumbing changes. No neural scorer is trained
  or shipped here; claiming one would exist is exactly the "fixture-demo
  vs. ship" conflation this repository's honesty rules forbid.
* **Unbounded `MERGE_BRANCHES` candidate discovery.** Only caller-proposed
  triples are checked — enumerating every possible `(base, left, right)`
  combination across an arbitrary trace is combinatorial and not
  attempted; this mirrors how `AstOperatorV1` argument domains are bounded
  (`max_combinations_per_operator`) rather than exhaustively searched.
* **`TRANSACTION_COMMIT`-containing branch segments for `MERGE_BRANCHES`.**
  Inherited directly from `sequence_merge.py`'s own excluded scope (DSH5-07)
  — a segment containing one is `_walk_branch_segment`-unsupported, the
  same conservative outcome DSH5-07 itself chose.
* **A live `enumerate_conversation_control_legal_set` `MERGE_BRANCHES`
  proof cache.** Each candidate is re-verified in full (including a real
  merge attempt) every time the legal set is built; no incremental/cached
  proof reuse is implemented, since none of this issue's acceptance
  criteria ask for it and premature caching would risk staleness bugs this
  issue has no test coverage to catch.

## Acceptance

* **100% selected-command membership and replay for supported actions** —
  every `apply_conversation_control_action` call dispatches to a function
  that is itself already replay-proven; `MERGE_BRANCHES`'s "legal" and
  "executed" paths are the identical code path, so membership and replay
  can never diverge.
* **Control and AST namespaces remain separate and fail closed** —
  `ConversationControlKind` is a standalone enum, never an `operator_id`
  string; nothing in this module registers into or reads from any
  `OperatorLibraryV1`.
* **STOP calibration reports Brier/ECE and premature/late stop** —
  `ConversationControlPolicyReportV1`.
* **Learned control is adopted only on causal held-out benefit;
  deterministic/API behavior remains available** —
  `deterministic_control_priority` is the shipped, fully-specified
  default; no learned arm has been adopted (none has been trained).

## Scope notes (deliberately deferred)

* **A real trained scorer and its held-out-benefit evaluation** — see
  "Coverage and what's excluded" above. This is the natural DSH5-09
  (adaptive router) follow-on's concern, which this issue's `Advancement`
  section names explicitly ("Supplies the safe control-plane mode required
  by the adaptive router").
* **Serializing `LegalControlActionV1` to/from a reserved-token codec** —
  `serialized` fields here follow the same human-readable
  `"CONTROL <KIND> <args>"` convention `serialize_operator_action` uses for
  AST operators, but no `deserialize_control_action` counterpart or
  reserved-token allocation is added; nothing in this issue's acceptance
  criteria requires round-tripping through a token vocabulary.

This work is compiler contract/unit fixtures exercised through the real
`openui` `DslPack` (parse/canonicalize/schema-oracle/round-trip authority
is the genuine pack backend, not a stub) plus pure-Python calibration
helpers. No train, eval, benchmark, matrix, checkpoint, model-card,
ship-gate, or model-quality claim is produced.
