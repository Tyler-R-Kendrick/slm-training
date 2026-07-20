# E606 — bounded semantic-plan component margin

Date: 2026-07-20
Status: completed and rejected for promotion

E606 adds a default-off score floor for still-required prompt-plan families.
With margin 2, each required family is placed two points above the best
currently legal component score. Existing remaining-count logic disables the
floor when planned cardinality is satisfied. The preregistered matched arm
keeps regular plan weight 4 and the complete E605 visible-contract policy.

The capped OOD `n=4` run completes normally and exactly matches E605 on every
headline metric: syntax 1.0, meaningful-v1 0.5, strict meaning-v2 0, fidelity
0.5917, validity 0.7550, structure 0.5756, component recall 0.6250, reward
0.8145, AST-node F1 0.6111, AST-edge F1 0.4643, and AgentV 0/1.

The intervention works locally but fails the intended end-to-end repair:

- dashboard emits `Button`, `Callout`, `Card`, and a second `Card`, satisfying
  the predicted family cardinality in the decode stream;
- all margin-stage winners remain the actual final legal token;
- canonicalization still returns only
  `root = TextContent(":ood.dash.status.body")`, so the planned bindings are
  unreachable and component recall remains 0.25;
- auth ordering changes to Input/Button/Input and adds a placeholder-spam
  failure reason without improving strict meaning-v2.

Reject margin 2 as a promotable policy. Keep the default-off lever as
diagnostic infrastructure: it proves family selection can be repaired without
an arbitrary absolute weight, but the next experiment must construct a
verified root over already-emitted planned bindings after coverage.

No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e606-semantic-plan-margin-20260720.json).
