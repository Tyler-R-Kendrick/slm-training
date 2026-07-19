# E510 — component-plan decode

E510 activates the rejected E505 checkpoint's already-trained component-plan
head at decode weight 4. The comparison retains E509's four OOD records,
honest slot-contract context and constraint, length-safe 160-token canvas,
default generation controls, and no DESIGN context.

The evaluation completed under its external 170-second cap and emitted
AgentEvals plus pinned AgentV evidence without execution errors.

| Policy | n | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | AST edge F1 | Strict v2 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E509 component plan off | 4 | 0.25 | 0.2583 | 0.2406 | 0.3333 | 0.6920 | 0.3389 | 0.0 | 0.0 | 0/1 |
| E510 component plan 4 | 4 | 0.50 | 0.6583 | 0.3446 | 0.3958 | 0.8405 | 0.4679 | 0.1625 | 0.0 | 0/1 |

The head applied 155 component-plan biases and changed seven legal component
choices. All headline OOD quality metrics improve, with no unconstrained
fallback. Remaining failures are narrower but important: missing required
components/placeholders, placeholder spam, and semantic-role mismatches keep
strict binding-aware meaningfulness and AgentV red.

## Decision

Retain component-plan weight 4 as the leading diagnostic policy and expand it
across held-out and adversarial suites. Do not promote the checkpoint until
strict semantic and AgentV gates improve. No checkpoint was created or synced.

Exact metrics:
[machine-readable record](iter-e510-component-plan-decode-20260719.json).
