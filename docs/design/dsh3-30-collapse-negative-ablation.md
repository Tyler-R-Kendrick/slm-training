# DSH3-30: collapse-negative ablation (SLM-405)

## Protocol

SLM-405 compares five predeclared, matched training arms: `no_negatives`,
`conflict_only`, `different_result_only`, `equal_mix`, and `curriculum_mix`.
Each arm must retain the same source-trace split, positive-row exposure,
examples, and budget.  The only intended difference is which replay-derived
negative labels the training objective consumes.

`collapse_conversation_trace` is the authority for every adjacent swap.  Its
reprojected `OperatorPolicyHardNegativeV1` evidence is frozen by
`freeze_collapse_negative_ablation`; it records a `CONFLICT` code or a
`DIFFERENT_RESULT` final-state digest outside the sanitized model input.  The
manifest fails closed if either type is absent from either source-disjoint
train/dev split.  It also rejects a source trace that appears in both splits.

This preflight intentionally does not run a label-only or one-type training
arm.  That would not answer the causal question.

## Local result — 2026-07-26

[`report.json`](dsh3-30-collapse-negative-ablation-20260726-local/report.json)
records a local CPU preflight using `train_text_only_01` and
`held_out_dual_card_01`, `steps=4`, and exact legal-set enumeration capped at
512 combinations per operator.  It found two replay-verified `CONFLICT` rows
and zero `DIFFERENT_RESULT` rows.  The frozen manifest therefore rejected the
matrix before any of the five SLM-405 arms ran:

| Train | Dev | Result |
| ---: | ---: | --- |
| conflict: 1; different-result: 0 | conflict: 1; different-result: 0 | reject — no matched different-result stratum |

The AgentEvals/AgentV bundle has four invariant checks passing and the
negative-ablation preflight intentionally failing.  This is a bounded local
wiring/data-availability result, not a checkpoint, human-rating, CAP2, or ship
claim.  No human rating gate is required or used.

## Next evidence required

Extend the canonical replay-backed operator corpus with source-disjoint,
replay-verified different-result examples and the named controls before
re-running.  The full rerun must report pairwise positive-versus-alternate
ranking, balanced outcome classification/calibration, replayed selected final
sequence, negative consumption, and the requested order-sensitivity strata.
