# DSH3-25 coverage-aware operator-policy objective

SLM-400 adds a pure, default-off learning contract above the DSH3-24
`OperatorPolicyRowV1` evidence. It does not change legal-set enumeration,
hard pruning, singleton forcing, decoder routing, or ship gates.

## Labels and denominators

`OperatorPolicyTargetV1` keeps compiler support
(`SUPPORTED`/`UNSUPPORTED`/`UNKNOWN`) orthogonal to target utility
(`POSITIVE`/`NEGATIVE`/`UNKNOWN`). An `UNKNOWN` compiler target is rejected if
it is labelled an ordinary negative. The public model boundary contains only
coverage and compiler-support tags: evaluator keys, utility labels, and the
complete shadow set are not model input.

`operator_policy_loss` exposes three comparison arms:

| Arm | Candidate denominator | Uncertainty treatment |
| --- | --- | --- |
| known-supported CE | compiler-`SUPPORTED` candidates | `UNKNOWN` excluded |
| PU-risk control | supported candidates with certified utility | unlabeled candidates excluded |
| explicit DEFER | DEFER class plus known candidates, or DEFER alone for partial/unknown rows | DEFER needs exact fallback success for credit |

Each return includes its actual denominator and complete/partial coverage
counts. Objective selection must compare held-out accepted-set mass, risk, and
UNKNOWN-mass ECE; training loss alone is not a selection criterion.

## Partial controls and inference

`build_controlled_partial_fixture` derives a public bounded prefix from a
known COMPLETE domain under a supplied truncation permutation. The complete
domain remains evaluator-only shadow truth. This supports unknown fractions
from zero through three quarters and order-permutation controls without
feeding the hidden remainder to a policy.

Inference routing is fail-closed:

| Condition | Route |
| --- | --- |
| complete, one known-supported action | `complete_singleton` |
| complete, multiple known-supported actions | `complete_ambiguous` |
| partial with a witness | `partial_witnessed` → DEFER/fallback |
| partial without a witness or repeated unbounded slot | `partial_unknown` → DEFER/fallback |
| timeout | `timeout` → DEFER/fallback |
| budget exhausted | `budget_exhausted` → DEFER/fallback |

Only compiler COMPLETE domains can return `complete_singleton`; partial
domains never bypass, even with one observed witness. A DEFER is credited only
when its downstream exact fallback succeeds. The compact report records
accepted-set mass, risk, UNKNOWN-mass ECE, false hard eliminations against the
hidden complete shadow, credited/uncredited DEFERs, and partial singleton
bypasses.

## Scope

This is a unit-tested schema/loss/routing contract, not a train, eval, matrix,
checkpoint, or ship claim. The next experiment must use the existing canonical
evaluation envelope and record a local bounded result under `docs/design/`.
