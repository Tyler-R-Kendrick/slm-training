# VAR3-02 (SLM-430): turn disposition as a first-class REPL-variant output

Date: 2026-07-26
Status: implemented; fixture/synthetic-scale capability evidence only
Scope: `dsl/operators/turn_disposition.py`, `models/turn_disposition_head.py`,
`evals/cap2_operator.py` (`score_disposition_predictions`),
`dsl/operators/conversation.py` (`TurnArtifactV1.disposition`), `levers.py`.
Honesty: **fixture_or_scratch / capability, not ship.** No promotion, gate
change, or production-readiness claim is made anywhere in this document.

## Decision

The REPL decode variant makes one classification decision per turn before it
commits anything. This issue makes that decision a first-class, typed output
-- `TurnDispositionKind` (`out_of_scope` / `clarify` / `answer` / `emit`) --
instead of leaving it implicit in whatever a downstream head happens to
produce.

## Precedence (I1 ordering)

`dsl/operators/turn_disposition.py`'s `derive_turn_disposition` resolves the
four dispositions in strict order; each earlier rule short-circuits every
later one:

1. **`out_of_scope`** -- the accepted legal set
   (`dsl.operators.legal_set.OperatorLegalSetV1`) is empty. Read directly off
   `legal_set.all_serialized_actions`; nothing else is ever consulted. There
   is no candidate for a learned score to even apply to, so this is proven,
   not just claimed: `test_out_of_scope_never_consults_score_provider`
   (`tests/test_dsl/test_turn_disposition.py`) passes a `score_provider` that
   raises `AssertionError` if called, with `is_state_query=True` as well, to
   prove out_of_scope outranks every other rule too.
2. **`answer`** -- a corpus/trace-derived `is_state_query` flag (never a
   model prediction) marks this turn as a question about existing state, not
   a mutation. Consumes replayed state and emits no op; recorded as a
   self-referential `COPY_STATE` turn (see "Trace representation" below).
3. **`emit` (forced)** -- `OperatorLegalSetV1.forced_action` (`COMPLETE`
   coverage, exactly one candidate) is the same I1 singleton bypass every
   other operator-policy surface in this repo honors
   (`models/operator_termination.py`'s `forced_singleton`,
   `control_actions.py`'s `deterministic_control_priority`). No score
   provider is consulted (`test_forced_singleton_bypasses_scoring`).
   A `PARTIAL`-coverage legal set that happens to enumerate only one
   candidate so far is **not** treated as forced -- the bounded scan cannot
   rule out more candidates beyond its budget, so it conservatively
   clarifies with that one candidate instead
   (`test_partial_coverage_single_candidate_is_conservatively_clarified`).
4. **`clarify` / `emit` (scored)** -- only now is `score_provider` consulted,
   for a top-1-vs-runner-up margin against the registered
   `levers.TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD`. Below the threshold
   returns `clarify` with the top `levers.TURN_DISPOSITION_CLARIFY_TOP_K`
   ranked candidates; at or above it returns `emit` with the top-ranked one.

## Registered levers (`src/slm_training/levers.py`)

| Lever | Value | Meaning |
| --- | --- | --- |
| `TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD` | `0.5` | Top-1/runner-up score margin below which a non-singleton legal set clarifies instead of emits. Defaulted conservative (more-clarify): a wide band means more ambiguous decisions surface a clarification instead of silently emitting a possibly-wrong op. |
| `TURN_DISPOSITION_CLARIFY_TOP_K` | `3` | How many ranked candidates a `clarify` disposition carries. |
| `TURN_DISPOSITION_WRONG_EMIT_PENALTY_RATIO` | `3.0` | Exact multiplier by which one wrong `emit` is penalized relative to one `clarify` abstention in the CAP2 composite (named constant, not buried in scoring code). |

## Disposition head (`models/turn_disposition_head.py`)

Mirrors `models/operator_termination.py`'s shape exactly: `TurnDispositionLabel`
is a **three-way** enum (`answer` / `clarify` / `emit`) that structurally
excludes `out_of_scope` -- the classifier's output space has no slot for it,
so a corpus builder cannot even mislabel an `out_of_scope` turn into this
classifier's training data by mistake. `turn_disposition_losses` is a
cross-entropy loss maskable on `forced_singleton` targets (same
`mask_forced` pattern as `operator_termination_losses`), and
`TurnDispositionReportV1.from_predictions` reuses
`flow.termination.brier_score`/`expected_calibration_error` rather than
reimplementing calibration math, exactly like `ConversationControlPolicyReportV1`
and `OperatorTerminationReportV1` already do.

No corpus builder or training run is added by this issue -- the head's loss
and report are the training-ready surface a future corpus-labeling issue
would target, tested here against synthetic tensors only
(`tests/test_models/test_turn_disposition_head.py`).

## CAP2 eval extension (`evals/cap2_operator.py`)

`score_disposition_predictions` reports `wrong_op_rate` and `abstention_rate`
**separately** -- never blended into one number that could hide either
regressing while the other improves (this repo's honest-ship-eval framing):
a wrong `emit` is a silently-wrong action a caller could apply outright; a
`clarify` abstention is visible and safe by construction. The composite:

```
composite_penalized_error_rate = (wrong_op_rate * ratio + abstention_rate) / (ratio + 1)
```

with `ratio = levers.TURN_DISPOSITION_WRONG_EMIT_PENALTY_RATIO = 3.0` -- one
wrong emit costs exactly 3x one clarify abstention, an explicit, registered
constant rather than an implicit equal weighting.

## Trace representation (`dsl/operators/conversation.py`)

`TurnArtifactV1` gained two optional, descriptive-only fields:
`disposition: str | None` and `disposition_candidates: tuple[str, ...]`.
Neither is consulted by `replay_conversation_trace` to decide state -- they
round-trip through the existing turn-comparison machinery exactly like every
other turn field, so no new replay branch was added.

`clarify` and `answer` are recorded as a **self-referential `COPY_STATE`
turn** (`copy_source_state_id == input_state_id`): I11 already establishes
that "a turn may re-materialize an earlier state's artifact on the current
branch," and copying the current state onto itself is exactly "one turn
later, same AST content, no mutation" -- precisely what both dispositions
mean. `append_clarify_turn`/`append_answer_turn`
(`dsl/operators/turn_disposition.py`) are thin wrappers over
`copy_conversation_state`; `out_of_scope` is deliberately **not**
recordable (excluded from `_RECORDABLE_DISPOSITIONS`) since it never
advances the trace with a turn. `emit` may optionally be recorded on a real
`AST_EDIT`/`TRANSACTION_COMMIT` turn via the same `disposition=` keyword on
`append_operator_turn`/`append_operator_transaction_turn`.

Proven by `tests/test_dsl/test_turn_disposition.py`:
- `test_clarify_turn_is_recorded_and_survives_replay` -- a `clarify` turn
  carrying two ranked candidates round-trips through
  `replay_conversation_trace` exactly, and the disposition/candidates
  survive `to_dict()` verbatim.
- `test_answer_turn_is_recorded_and_survives_replay` -- same, for `answer`.
- `test_clarify_and_answer_dispositions_require_a_self_referential_copy` --
  `TurnArtifactV1` fails closed on an `out_of_scope` disposition string, a
  `clarify` with no candidates, an `emit` on a non-mutating operation, and
  stray candidates without a `clarify` disposition.

## Fixture measurement: disposition-on vs. disposition-off

**Honesty: fixture_or_scratch, capability class -- not a ship claim.**
`scripts/run_var3_02_turn_disposition_fixture.py` builds one deterministic,
seeded (`seed=430`) synthetic corpus of 200 turns covering every disposition
branch (10% out_of_scope, 15% state-query, 15% forced-singleton, 30%
ambiguous-margin, 30% confident-margin-but-imperfect-scorer) and scores two
matched-budget arms over the **identical** corpus and identical (imperfect)
candidate scores:

- `disposition_off` -- the pre-VAR3-02 baseline: always attempts to `emit`
  the top-scored candidate, with no `out_of_scope`/`answer`/clarify-margin
  concept at all. Forcing an emit against an empty legal set or a
  state-query turn is definitionally wrong.
- `disposition_on` -- `derive_turn_disposition` with the registered levers
  above.

Measured via `python scripts/run_var3_02_turn_disposition_fixture.py`
(committed result: `docs/design/var3-02-turn-disposition-20260726.json`):

| Arm | case_count | wrong_op_rate | abstention_rate | composite_penalized_error_rate |
| --- | ---: | ---: | ---: | ---: |
| `disposition_off` | 200 | **0.555** (111/200) | 0.000 (0/200) | 0.416 |
| `disposition_on` | 200 | **0.055** (11/200) | 0.285 (57/200) | 0.112 |

Reported honestly, both directions: `disposition_on` trades a nonzero
`abstention_rate` (28.5% of turns clarify instead of committing) for a large
drop in `wrong_op_rate` (55.5% -> 5.5%) on this synthetic corpus, and the
penalty-weighted composite improves accordingly (0.416 -> 0.112). This is
exactly the intended trade this issue's precedence order encodes -- it is
**not** evidence that either number is acceptable in production, only that
the derivation logic and CAP2 scoring extension behave as designed on a
corpus built to exercise every branch.

## Verification run in this session

- `python -m pytest tests/test_dsl/test_turn_disposition.py
  tests/test_models/test_turn_disposition_head.py
  tests/test_evals/test_cap2_operator_turn_disposition.py -q` -- 23 passed.
- `python -m pytest tests/test_dsl/ -q` and `python -m pytest tests/test_models/ -q`
  (full neighboring suites) -- see PR/session report for exact counts.
- `python -m scripts.verify_decode_invariants`, `python -m
  scripts.verify_version_stamps --check`, `python -m scripts.repo_policy` --
  see PR/session report.

## Explicitly out of scope (per issue)

- Natural-language generation for `clarify`/`answer` turns -- both carry
  only structured candidates/state, never prose.
- Any change to legal-set computation itself
  (`dsl/operators/legal_set.py`) -- this issue only consumes
  `OperatorLegalSetV1`.
- Any promotion, ship-gate, or production-readiness claim -- this is
  `capability`-class, fixture/synthetic-scale evidence only.

## Environment note (unrelated to this issue's code)

The sandbox used for this session initially failed almost every `dsl/`
test with `OperatorAuthorityError: pack static/schema oracle rejected
source` -- root cause was (1) `src/apps/openui_bridge` node dependencies not
installed (`npm ci`), and (2) a globally-set `NODE_OPTIONS=--import tsx`
environment variable that the bridge subprocess rejects
(`node: --import tsx is not allowed in NODE_OPTIONS`). Both are pre-existing
environment setup issues reproducible on a clean checkout with none of this
issue's changes applied, not regressions introduced here. Tests in this
session were run with `env -u NODE_OPTIONS` after `npm ci` in
`src/apps/openui_bridge`.
