# E515 — focal-loss decomposition

E515 repeats E513 from the same E396 parent with the same E500 primary corpus,
50% E357 replay, seed, 5,000-target-token budget, slot-component loss 4, honest
slot-contract context, and all other settings. The only changed training lever
is focal gamma `2→0`.

The CPU HF-context run stopped normally after 101 steps and 105.8 seconds,
inside `max_wall_minutes=3` and the external 170-second cap. Its serving
checkpoint SHA-256 is
`97f2e426604e3956f2791398a608b967937ebf548fa7cae0ef59dde324721c1b`;
checkpoint, full state, metadata, and references are uploaded and verified at
`hf://buckets/TKendrick/OpenUI/checkpoints/e515-e396-e500-replay050-slotrole4-focal0-r1-5k`.

E516 evaluates that exact checkpoint on the matched four-record OOD suite and
publishes AgentEvals plus pinned AgentV evidence.

| Arm | Focal gamma | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | AST edge F1 | Strict v2 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E510 leading policy | 0 | 0.50 | 0.6583 | 0.3446 | 0.3958 | 0.8405 | 0.4679 | 0.1625 | 0.00 | 0/1 |
| E513 slot loss 4 | 2 | 0.00 | 0.4917 | 0.2750 | 0.2083 | 0.7695 | 0.3500 | 0.0625 | 0.00 | 0/1 |
| E515 slot loss 4 | 0 | 0.25 | 0.6583 | 0.3213 | 0.2708 | 0.8270 | 0.4292 | 0.0625 | 0.00 | 0/1 |

## Decision

Focal gamma 2 is harmful in this recipe. Resetting it to zero recovers most of
E513's regression, but slot-component loss 4 still trails the E510 checkpoint
and does not clear strict meaning or AgentV. Keep focal gamma zero, reject this
checkpoint for promotion, and return slot-component loss to 1 while improving
role-labeled training coverage instead of increasing objective scale.

Exact recipe, provenance, metrics, and gate outcome:
[machine-readable record](iter-e515-focal-loss-decomposition-20260719.json).
