# DSH3-13 frozen CAP2 operator evaluation (SLM-381)

Date: 2026-07-23
Status: frozen symbolic fixture contract passed
Scope: CAP2 evaluation definition and anti-cheat controls; no model or ship claim

## Decision

CAP2 now has one hash-addressed symbolic evaluation suite generated through the
canonical operator-corpus and replay contracts. The suite is frozen by:

- the two selected `held_out` source-record IDs and their canonical content
  fingerprint;
- the generated operator-corpus fingerprint;
- the complete ordered case payload hash;
- versioned raw-rate, accepted-mass, and Wilson-lower-bound thresholds; and
- normalized component versions for every operator/replay dependency.

Any source, generated-gold, case, stratum, threshold, or ordering change
invalidates the suite hash. Gold transitions and intermediate states are
admitted only after canonical operator application and immutable conversation
replay. The suite contains no natural-language rows because SLM-379/CERT_CAP1
is unavailable.

## Frozen surface

Suite `cap2_operator_v1` contains 20 cases:

| Stratum | n |
| --- | ---: |
| Held-out single transitions | 4 |
| Held-out multi-turn compositions | 2 |
| Branch isolation | 2 |
| Undo/redo exact state identity | 2 |
| Sequential-versus-collapsed equality | 2 |
| Reordered noncommuting conflicts | 2 |
| Stale-reference refusal | 2 |
| Opaque-marker permutation | 2 |
| Merge replay / typed conflict contracts | 1 / 1 |

Transition rows score accepted legal-action mass, operator ID, typed-argument
fingerprint, exact canonical AST, `ActionEffect` fingerprint, locality,
unintended edits, and final state. Other rows score their exact history,
ordering, conflict, permutation, or merge contract. Every row also enforces an
anti-bloat operator-count ceiling and applicable empty/trivial-output checks.

The merge rows freeze the already implemented `branch_merge/v1` replay and
`merge_conflict/v1` typed-refusal contracts; this fixture does not claim a new
learned merge evaluation. CAP0 retention, CAP1 retention, and current strict
model metrics are explicit unavailable diagnostics until a learned checkpoint
and CERT_CAP1 exist.

## Confidence-bound gate

The frozen gate requires:

- mean accepted legal-action mass at least `0.9`;
- every applicable binary check to have raw rate `1.0`;
- every dimension's 95% Wilson lower bound at least `0.2`; and
- the overall case-pass 95% Wilson lower bound at least `0.75`.

The low per-dimension bound acknowledges fixture-sized denominators; it is a
contract-regression threshold, not evidence of model quality. Later CAP2 model
claims must use powered suites and the repository's ordinary strict/AgentV
ship policy.

## Fixture result

The final CPU run used no checkpoint, no model backend, zero train steps, and
completed in 64.38 seconds under the three-minute cap. Peak traced memory was
16,952,098 bytes.

| Policy | Exact cases | 95% Wilson lower | Gate |
| --- | ---: | ---: | --- |
| Replay-authoritative oracle | 20 / 20 | 0.8389 | pass |
| Unchanged input | 0 / 20 | 0.0000 | fail |
| Generic valid AST | 0 / 20 | 0.0000 | fail |
| Constant operator/AST | 1 / 20 | 0.0089 | fail |

Thus a generic valid program, unchanged input, or memorized constant operator
cannot pass this fixture contract. This does not show that any learned model
can pass it.

The AgentEvals spec and pinned AgentV result bundle are committed under
[`dsh3-13-cap2-operator-eval-20260723/`](dsh3-13-cap2-operator-eval-20260723/).
AgentV passed 6/6 evidence-integrity cases with mean score 1.0 and zero
execution errors. The complete suite hash is
`16f210786bac7fd5f5edb64d13888c3cc7d634330a81b5065150e7a41fcb1d4d`;
the generated operator-corpus fingerprint is
`5ee0d27141a3fa72be35bedbdec347f97f513c0e7af672ca4be580e5b982682e`.

## Telemetry and honesty boundary

The prediction schema reserves and aggregates active nodes, node passes,
remask phases, model/compiler/verifier calls, latency, and peak memory. These
fields are instrumentation for later systems comparisons; this fixture's
oracle uses no model and cannot support latency or efficiency claims.

No checkpoint was created, so the model card and README checkpoint summary do
not change. No ship gates were run. The passing AgentV bundle proves that the
evaluation evidence is complete and internally consistent, not that CAP2 is
certified.

## Research lineage

[Saha and Kanewala, 2018](https://arxiv.org/abs/1802.07361) motivates
systematic source/follow-up construction for metamorphic fault detection.
[Tarlow et al., 2019](https://arxiv.org/abs/1911.01205) motivates predicting
precise AST diffs with pointer-like code locations. DSH3-13 is an adapted
repository evaluation contract: neither paper defines these OpenUI metrics,
history/merge/collapse cases, anti-cheat controls, thresholds, or AgentV
envelope, and no paper result is reproduced.
