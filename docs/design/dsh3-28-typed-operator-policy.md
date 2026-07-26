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

## Remaining evaluation scope

This is the reusable scored-policy boundary and its unit proof, not a
promotion or ship claim. The issue still requires the matched five-head CAP2
matrix on a current suite whose legal-set coverage supplies COMPLETE ambiguous
rows, with per-decision causal-change denominators and full-generation/
serialized baselines. No checkpoint is created by this step.

The legacy CAP2 v1 generator currently detects drift after later
selector/effect-contract evolution. That is an in-repository compatibility
problem to resolve with a current-surface evaluation while keeping the v1
evidence immutable; it is not a claim that the historical gate passed.

## Regression coverage

`tests/test_harnesses/experiments/test_typed_operator_policy.py` proves that
the scorer selects only live typed rows, COMPLETE singleton decisions execute
zero scorer forwards, and PARTIAL inputs defer. The policy-view/corpus tests
prove that canonical persistence keeps evaluator-only label joins intact.
