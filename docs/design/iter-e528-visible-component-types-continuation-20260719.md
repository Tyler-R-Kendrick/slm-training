# E528 — visible component-types continuation

E528 tests whether E527's type-only component contract keeps the useful part of
E525's inventory signal without exposing exact counts. The parent, 50% E357
replay, 5k-token budget, honest context authority, objective weights, seed, and
evaluator are fixed. E527 has the same 244 IDs and OpenUI targets as E521 and
E524, so the only primary-corpus difference is the weaker contract text.

## Train, persistence, and checkpoint

The clean-source CPU HF-context run completes 99 steps / 5,059 target tokens in
146.75 seconds under `max_wall_minutes=3`. Automatic HF Bucket sync and
verification passed, and an independent listing confirmed the complete
nine-file bundle at
`hf://buckets/TKendrick/OpenUI/checkpoints/e528-e396-e527-replay050-slotrole1-honest-context-r1-5k/`.
The serving SHA is `6a2180d7…306976d5` and the full-state SHA is
`4a70677f…85aa7`.

## Matched OOD result

| Metric | E523 / E522 slot contract | E526 / E525 count contract | E529 / E528 type contract |
| --- | ---: | ---: | ---: |
| Meaningful | 0.0000 | 0.0000 | 0.2500 |
| Placeholder fidelity | 0.8667 | 0.4667 | 0.5500 |
| Structure | 0.1955 | 0.1452 | 0.1136 |
| Component recall | 0.2708 | 0.4167 | 0.3542 |
| Reward | 0.2093 | 0.1668 | 0.5778 |
| AST node F1 | 0.3437 | 0.3041 | 0.2270 |
| AST edge F1 | 0.1007 | 0.0774 | 0.0801 |
| Strict binding-aware meaning | 0.0000 | 0.0000 | 0.0000 |
| AgentV | 0/1 | 0/1 | 0/1 |

The n=4 diagnostic retains syntax and recovers v1 meaningful rate and reward.
Against E525, fidelity also rises by 0.0833 while recall falls by 0.0625.
However, hierarchy does not improve: structure and AST node F1 regress, two
outputs have low component recall, one is trivial, and three miss required
prompt components. The large latency change is an unreplicated runtime
variance, not a performance claim.

## Decision

Reject E528. Type-only conditioning is a useful diagnostic because it recovers
some semantic behavior without exact counts, but strict meaning and AgentV are
still zero. The next intervention should supervise semantic roles and
reference-graph construction without revealing gold inventory or weakening
the gates. Machine-readable evidence is in
[the E528 JSON](iter-e528-visible-component-types-continuation-20260719.json).
