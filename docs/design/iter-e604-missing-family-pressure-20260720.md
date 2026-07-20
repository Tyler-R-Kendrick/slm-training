# E604 — missing-family plan-pressure ladder

Date: 2026-07-20
Status: completed and rejected; not promotable or ship

E604 preregistered regular semantic-plan weights 16 and 32 with the
first-binding seed disabled. Both arms use the exact E603 matched visible
contract policy and checkpoint. The acceptance rule required reachable planned
family coverage or strict meaning-v2 improvement without placeholder/reward
regression or duplicate-family spam.

Both capped OOD `n=4` arms complete normally and are aggregate-identical:
syntax 1.0, meaningful-v1 0.5, strict meaning-v2 0, fidelity 0.5917, validity
0.7550, structure 0.5756, component recall 0.6250, reward 0.8145, AST-node F1
0.6111, AST-edge F1 0.4643, and AgentV 0/1.

The apparent gain over E603 structure 0.5169 is not the intended repair:

- dashboard remains a one-node root with component recall 0.25. Weight 16
  renders `TextContent`; weight 32 only flips it to `Button`. Both omit the
  required Card and Callout families;
- gallery remains a one-node `TextContent` root with recall 0.25 in both arms;
- modal is unchanged;
- auth reaches exact AST/ref-graph shape by emitting two `Input` bindings, one
  of which duplicates/misbinds the name placeholder. Strict meaning-v2 still
  rejects it.

The result rejects scalar plan pressure as the repair. Higher weights cannot
turn locally preferred missing-family sections into a reachable root graph and
instead amplify duplicate structure. Keep regular plan weight 4 as the scratch
baseline. The next lever should couple missing-family section commitment to
root reachability rather than increase a global score.

No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e604-missing-family-pressure-20260720.json).
