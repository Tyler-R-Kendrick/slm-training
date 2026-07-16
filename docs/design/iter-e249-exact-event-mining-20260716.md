# E249 exact-event mining prerequisite (2026-07-16)

**Outcome:** the production strict compiler-tree path now emits replayable exact-state
decision events, and a 2,035-event immutable corpus is committed for future E249-E251
runs. No training or quality evaluation ran, so this result makes no model-quality or
ship claim.

## Recipe and evidence

- Device: CPU; seed: `0`; policy: strict compiler tree; target kind: `document`.
- Parent policy SHA: `7a9be4a665e216d7f7e73883ad74ad972bbf30846896d0c29188d6482f5b093a`.
- Source-record fingerprint: `9f72d85b6cc7118e0f69e010d0debdd2b40ede514e03178dded8e164daaae9bb`.
- Decode-config hash: `8c2a2ae5cb5c4ad0ab44de74a172fc2c887fbdc7e5e7e20ede953184a0c3fe5b`.
- Full trace: `f8186cf6d911344416a8b6ee3a9a2f71`.
- Machine-readable record: [`iter-e249-exact-event-mining-20260716.json`](iter-e249-exact-event-mining-20260716.json).

The full pass accepted all 65 document records and observed 2,368 production
compiler commits. Mining produced 2,035 unique constraint-shadow events in 65 prompt
groups: 1,716 events / 54 groups in train and 319 events / 11 groups held out. The
committed corpus fingerprint is
`8d9b18827e47ba24c67c6482bbd5d36308cd648acd4cad61b2eeb1428c90d97c`.

The first eight-row diagnostic (`f5a341d29dc48083a202cdcf851e9b00`) was invalid:
those rows targeted statement/expression outputs, so the production generator
correctly did not enter document compiler-tree decoding and emitted no events. The
collector now fails closed when strict-tree collection includes a non-document row.
An eight-document control (`e46fa63f138aeb4edd37b05817ed7f76`) then accepted 8/8,
observed 156 compiler commits, and mined 132 events before the full pass.

## What changed

Compiler-tree branch recording is attached to the production decoder and captures
the exact context, pre-decision canvas, legal support, selected token, and raw
full-vocabulary argmax. Singleton legal-token states remain deterministic and are
not converted into trainable decisions. The strict policy is shared by matrix and
trajectory entrypoints rather than duplicated as experiment-specific literals.

The corpus has an identity-homogeneous manifest and is exposed by the dashboard's
Training Data page in both compiled and interpreted modes. Direct browser checks
showed the same eight columns and one corpus row in both modes, with no console
errors; the API returned the same 2,035-event committed record.

## Honesty boundary and next action

Every event is a constraint shadow: the unconstrained raw argmax was illegal while
the selected compiler token was legal. This certifies lexical legality only. It is
not evidence that the constrained token is semantically better, and there are no
counterfactual or set-valued events. E249-E251 may use this corpus as a matched
grammar decision experiment; E252-E254 remain blocked by their explicit evidence
gate. Immediately before E249 training, fetch and reconcile latest `origin/main`,
then prove a clean worktree and zero commits behind.
