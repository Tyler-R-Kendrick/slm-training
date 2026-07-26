# DSH3-28 typed dynamic operator policy

SLM-403 starts the first default-off trained policy path over the frozen
`OperatorPolicyInputV1` boundary. It composes, rather than replaces,
DSH3-22's sanitized view, DSH3-23's ragged typed encoder, DSH3-24's
collapse-derived rows, DSH3-25's coverage-aware objective, and DSH3-26's
termination routing.

## Implemented boundary

`typed_operator_policy.py` contains `TypedOperatorPolicyScorer`: a typed
ragged action scorer plus a typed action-conditioned candidate scorer. It
accepts only an `OperatorPolicyInputV1`; targets remain in
`TypedOperatorPolicyExampleV1` as evaluator-only row indices. The persisted
payload is rehydrated through `operator_policy_input_from_dict()`, so training
uses the same canonical action/reference order that DSH3-24 labels validate.

The initial path is default-off and has three non-negotiable inference rules:

* COMPLETE singleton decisions bypass the learned scorer (`model_forwards=0`).
* COMPLETE ambiguous decisions may select only an existing action row and an
  existing candidate row in its compiler-provided argument domain.
* PARTIAL decisions defer without a score, a forced action, or a hard prune.

The corpus builder now remaps accepted and hard-negative action/reference
labels through `OperatorPolicyInputV1.canonical_row_maps()` before persistence.
That prevents canonical evidence ordering from silently changing which legal
candidate a label names.

## Local bounded result (2026-07-26)

`dsh3-28-typed-operator-policy-20260726/report.json` records the first
bounded CPU current-surface probe, including AgentEvals JSONL and a pinned
AgentV result bundle. It used four train policy rows from two local roots and
two held-out policy rows from one local root, with an eight-combination cap.
All six rows were PARTIAL; consequently there were zero COMPLETE training
rows. The enabled and shuffled-label arms were deliberately skipped rather
than treating UNKNOWN candidates as negatives or force-emitting a choice.

The zero and random controls both forced zero PARTIAL choices; singleton
forwards were zero and the prediction replay matched. AgentEvals records those
three structural invariants as passing, but the experiment verdict is
**reject**: this bounded probe cannot test the trained-policy hypothesis. It
does not satisfy the issue's causal-effect acceptance criteria, makes no ship
claim, creates no checkpoint, and does not promote DSH5.

The corpus re-projection now receives the same explicit combination cap as the
canonical trace generator. Previously it could silently fall back to the
unbounded enum default while re-enumerating a collapsed state. That was a
local resource-safety and reproducibility defect, not evidence of model
quality.

## Bounded COMPLETE-row control result (2026-07-26)

[`dsh3-28-typed-operator-policy-20260726-cap512/report.json`](dsh3-28-typed-operator-policy-20260726-cap512/report.json)
and its [`summary`](dsh3-28-typed-operator-policy-20260726-cap512/summary.md)
record a completed CPU-only, four-step local preflight. It uses one explicit
train root (`train_text_only_01`) and one held-out root (`held_out_input_01`)
at an exact 512-combination cap. The admitted source snapshot repeats record
IDs, so the runner chooses the first source-order instance deterministically
rather than treating duplicates as independent roots.

Both train rows and both held-out rows were COMPLETE. The three AgentEvals
structural assertions passed under pinned AgentV: live typed-domain selection,
zero forwards for singleton routing, and replay/control denominator integrity.
The learned arm matched zero control (0/2 choice changes); it differed from
the random control on 1/2 rows, correctly. This is a measured negative causal
preflight: it does not meet the required >=5% enabled-versus-disabled choice
change or a held-out improvement, is not a CAP2 ship evaluation, creates no
checkpoint, and does not clear SLM-403.

## Five-family bounded control result (2026-07-26)

[`dsh3-28-typed-operator-policy-20260726-cap512-fiveheads/report.json`](dsh3-28-typed-operator-policy-20260726-cap512-fiveheads/report.json)
records the required local-flat, ternary-ECOC, factorized, independent-set, and
recurrent-set policy heads under identical CPU-only data, seed, four-step
budget, and controls. The run retained exactly two COMPLETE train rows and two
COMPLETE held-out rows. Each head ran enabled, weight-zero, shuffled-label, and
random controls; all three AgentEvals assertions passed under pinned AgentV.

Every enabled head made **0/2** held-out choice changes relative to its
weight-zero control. Each had one correct difference from random, but that is
not a causal enabled-versus-disabled effect. The runner therefore records the
preregistered stop-rule verdict **reject**: no family caused a beneficial
held-out change at this tiny local scale. This keeps all five implementation
families and their control evidence, but advances neither DSH5 nor a ship
claim. It also does not satisfy SLM-403's full CAP2 acceptance: the normal
eval ModelPlugin side channel is now integrated, but the frozen CAP2 suite
still needs its current-surface reconciliation.

The review-corrected rerun retains all 20 `(head_family, arm)` values in the
two structural AgentV cases (instead of collapsing them by arm), uses a
genuinely multiplicative factorized head, and records the current clean-source
stamp (`harness.experiments.typed_operator_policy` v8 and `model.quantization`
v7). Its three AgentEvals assertions pass; the negative causal conclusion is
unchanged.

## Remaining evaluation scope

This is the reusable scored-policy boundary and its unit proof, not a
promotion or ship claim. `TypedOperatorPolicyEvidencePlugin` now composes a
policy decision with the existing `ModelPlugin` generation-evidence channel:
the delegate remains the sole owner of materialized OpenUI, while normal
evaluation records a request-aligned `typed_operator_policy_evidence/v1` side
channel. Evidence-count mismatch fails closed. The integration regression
proves the unchanged delegate output receives meaningful-program scoring.

The issue still requires the matched five-head CAP2 matrix on a current suite
whose legal-set coverage supplies COMPLETE ambiguous rows, with per-decision
causal-change denominators and full-generation/serialized baselines. No
checkpoint is created by this step.

The immutable DSH3-13 `cap2_operator_v1` manifest still records its original
`5ee0…268e` corpus and `16f2…1d4d` suite hashes. Later turn-serialization and
node-flow effect changes alter action identities (including two selected
dual-card actions), so v1 cannot honestly be relabeled as current. The new
`cap2_operator_v2` manifest binds those same held-out source IDs to the live
`a922…97cf` corpus and `e80f…4491` suite hashes while retaining the v1 file
unchanged.

[`dsh3-28-cap2-operator-v2-20260726/report.json`](dsh3-28-cap2-operator-v2-20260726/report.json)
records its local CPU, zero-step fixture replay: all 20 cases replay through
the current compiler, the oracle clears the contract, all three degenerate
controls fail, and AgentV passes 6/6. This is a current-surface fixture
contract only—not a learned-policy result or ship claim—and it creates no
checkpoint. The matched five-head learned CAP2 matrix remains the outstanding
SLM-403 work.

## Regression coverage

`tests/test_harnesses/experiments/test_typed_operator_policy.py` proves that
the scorer selects only live typed rows, COMPLETE singleton decisions execute
zero scorer forwards, and PARTIAL inputs defer. The policy-view/corpus tests
prove that canonical persistence keeps evaluator-only label joins intact.

## CAP2 v2 current-surface stop-rule result (2026-07-26)

[`dsh3-28-typed-operator-policy-20260726-cap2-v2-fiveheads/report.json`](dsh3-28-typed-operator-policy-20260726-cap2-v2-fiveheads/report.json)
records the five-head CPU-only matrix against the executable current-surface
CAP2 v2 fixture. The two train and four held-out policy rows were all
PARTIAL at the fixture's immutable 32-combination cap. The harness therefore
did not force a selected action or train enabled/shuffled-label arms; each
head's zero/random controls materialized four evaluator-side selections via
fresh legal-set enumeration and compiler application. The remaining 16
fixture cases replayed through the fixture oracle, while all four deferred transition
rows failed closed, yielding 16/20 CAP2 cases rather than a false pass.

AgentV passed its three structural assertions. No enabled-versus-zero causal
denominator exists because no COMPLETE training row exists, so the
preregistered verdict is **reject**. This closes the trained-policy hypothesis
at the current CAP2 v2 coverage budget; it is neither a ship claim nor evidence
that a larger complete-coverage corpus would fail.
