# DSH5-10: the replay-preference -> `TypedOperatorPolicyScorer` adapter gap

**Honesty:** investigation / design note only. No code changed, no training or
evaluation run executed, no `version_stamp` bump (nothing watched by
`versions.json` was touched). Not a ship or readiness claim.

## Why this doc

DSH5-10's row-extraction slices (`docs/design/dsh5-10-replay-preference-rows.md`,
v1-v7) and PR #1129's fixture-scale ablation both explicitly leave "real
SFT/preference training against `TypedOperatorPolicyScorer`" as open scope.
PR #1131 ("convert replay preference rows to `PreferencePair`") found a
structural mismatch and stopped there, without a design for what actually
closes the gap. This scheduled session picked up that thread, per the
repeated "next steps" notes in the batch-size and joint-sweep smoke-loop
PRs (#1130, #1132) pointing here instead of further fixture-loop variance
checks. Given `MAX_RUN_MINUTES=3` (`src/slm_training/levers.py:20`), wiring
a real trained head in one bounded autonomous session would be rushing a
training claim — this doc scopes the actual adapter work so a future
session (with the time budget it needs) doesn't have to re-derive it.

## The two-part mismatch

### Part 1 (already documented by PR #1131): reward shape

`PreferencePair` (`chosen`/`rejected`/`chosen_score`/`rejected_score`) is
built for pairs of full OpenUI program renderings scored by
`composite_reward` (grammar/placeholder/layout). Replay-preference
`chosen`/`rejected` are bare action tokens (`"undo"`, `"checkout:<state>"`,
serialized `OPERATOR ...` calls) — scoring a token like `"undo"` through
`composite_reward` would silently manufacture a meaningless `0.0`. PR #1131
correctly leaves `chosen_score`/`rejected_score` at the dataclass default
and marks the schema `operator_replay_preference_pair/v1` to keep this
honest, but wires nothing further.

### Part 2 (new finding this session): the action-space schema gap

`TypedOperatorPolicyExampleV1` (`src/slm_training/harnesses/experiments/typed_operator_policy.py:180-217`)
requires an `OperatorPolicyInputV1` view (`src/slm_training/models/operator_policy_view.py:281-296`)
whose `action_rows: tuple[OperatorActionViewV1, ...]` are densely
row-indexed, plus an `accepted_action_row` index into that tuple.

`OperatorActionViewV1` (`operator_policy_view.py:242-260`) is typed
specifically for **compiler-registered operator invocations**: it requires
`operator_id`, `operator_version`, `verdict: OperatorSupportVerdict`,
`coverage: LegalSetCoverage`, `effect_signature`, and `argument_slots`.
There is no field, variant, or documented convention for representing a
history-control action (`undo`, `redo:<state>`, `checkout:<state>`, a merge
token) as a row in this view — those actions have no `operator_id` in the
compiler's operator registry and no `OperatorSupportVerdict`, because they
were never routed through the operator-legality compiler at all (see
`_available_history_actions` in `src/slm_training/dsl/operators/replay_preference.py`,
which enumerates them as string tokens alongside, not instead of, the real
`OperatorActionViewV1`-shaped legal set).

**This means the mismatch is not merely "no reward for these tokens" (part
1) — for 6 of the 7 confirmed `ReplayPreferenceRelation` members, the
*chosen action itself* cannot be expressed as a row in
`TypedOperatorPolicyExampleV1.view.action_rows` at all**, regardless of
reward handling:

| Relation (7 confirmed, `replay_preference.py:84-91`) | `chosen_action` shape | Maps to an `OperatorActionViewV1` row? |
| --- | --- | --- |
| `EDIT_THEN_UNDO` | literal `"undo"` (line 287) | No — history control |
| `UNDO_THEN_REDO` | `redo:<child>` token (line 315) | No — history control |
| `PARTIAL_ROLLBACK` | literal `"undo"` (line 342) | No — history control |
| `CHECKOUT_ANOTHER_STATE` | `checkout:<state>` token | No — history control |
| `FORK_THEN_CHOOSE_ONE_BRANCH` | `checkout:<state>` token | No — history control |
| `MERGE_SUCCESS` | merge token (line 541) | No — history control |
| `PRONOUN_FOCUS_FOLLOWUP` | `chosen_match.serialized`, drawn directly from `entry.legal_actions` (lines 383-407) | **Yes** — this is the one relation where both `chosen`/`rejected` are real operator applications from the compiler's own legal-action set |

Only `PRONOUN_FOCUS_FOLLOWUP` rows are, as-is, candidates for
`TypedOperatorPolicyExampleV1` (their `entry.legal_actions` members are
already `OperatorActionViewV1`-compatible legal-set entries — this still
needs a row-index join adapter, but no new schema). The other six relations
need a design decision before any of them can train a
`TypedOperatorPolicyScorer` head.

## Two candidate designs (neither implemented here)

**A. Extend `OperatorPolicyInputV1`/`OperatorActionViewV1` with a
history-control action-row variant.** Add a discriminated union or a
sibling `HistoryControlActionViewV1` (its own `row`/`schema`, no
`operator_id`/`verdict`/`coverage` since those don't apply to
undo/redo/checkout/merge) so `action_rows` can mix operator and
history-control candidates in one legal set the way
`_available_history_actions` already does at the string level. Pro: one
unified policy head eventually scores "should the model invoke operator X,
or undo, or checkout Y" as one decision, matching what the replay data
actually represents. Con: a real schema change to a `frozen(True)`
dataclass with `validate_no_forbidden_fields` and row-density invariants
(`operator_policy_view.py:299-311`) — needs its own version-stamped design
and test pass before any training code touches it, and every existing
`OperatorPolicyInputV1` producer/consumer would need auditing for the new
variant.

**B. A separate scorer/head for the history-control decision space,
distinct from `TypedOperatorPolicyScorer`.** Keep `OperatorActionViewV1`
scoped to real operators (its current, working contract); build a small
sibling head that scores over the history-control action set
(`undo`/`redo:<child>`/`checkout:<state>`/merge tokens) using whatever
state-graph features are available at `input_state_id` (already
fingerprinted via `legal_set_fingerprint`). Pro: no change to the existing,
tested `OperatorPolicyInputV1` contract; 6 of 7 relations get a real,
scoped target immediately. Con: two separate heads means two separate
training/eval loops, and no single joint "what should the model do next"
score across both action families — exactly the four-baseline comparison
DSH5-10's acceptance criteria ask for would need to combine both heads'
outputs, not just one.

## Recommendation for the next session with budget for this

Prototype **(B) first**: it needs no schema change to already-tested code,
unblocks 6 of the 7 relations immediately, and is the smaller, safer slice.
(A) is the more architecturally correct long-term answer if a single joint
policy over operators-and-history-controls is ever wanted, but should be a
follow-on decision made with the DSH3/DSH5 policy-head owners, not decided
unilaterally by one autonomous docs-only session extending an existing
frozen dataclass's invariants.

Concretely, the smallest real next step is: define a
`HistoryControlPolicyInputV1`-shaped extraction from
`OperatorReplayPreferenceRowV1` rows (reusing `legal_set_fingerprint` and
`_available_history_actions`'s already-computed candidate set), a minimal
scorer over it, and a single powered training run against the **real**
(non-synthetic) corpus PR #1129 explicitly flagged as still missing — not
the 8-session/40-row fixture that produced its `no_benefit_fixture_scale`
ceiling-effect result.

## Explicitly out of scope for this doc

No code, no training run, no held-out evaluation, no `PreferencePair`
generation, no schema change to `OperatorPolicyInputV1`. This is a design
note narrowing what the next real implementation session needs to decide
before writing code, per this repo's honesty rules — an unimplemented
adapter is not evidence of anything and makes no readiness claim.
