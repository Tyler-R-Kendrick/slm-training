# E522 — visible slot-contract continuation

E522 isolates E521’s prompt-visible inventory under the matched E519
continuation recipe. The parent, 50% E357 replay, 5k-token budget, honest
context authority, slot loss 1, component-plan loss 4, decode weights, seed,
and evaluator are held fixed; only the primary corpus changes from E500 to
E521.

## Train and checkpoint

The clean-source CPU HF-context run completes 99 steps / 5,059 target tokens in
120.69 seconds under `max_wall_minutes=3`. Automatic upload and independent
bucket listing verify the serving checkpoint SHA
`97cb10f4…bf420ce`, full-state SHA `e0eafca2…5d3b0e8`, tokenizers, metadata,
references, and train summary at
`hf://buckets/TKendrick/OpenUI/checkpoints/e522-e396-e521-replay050-slotrole1-honest-context-r2-5k/`.

Two setup attempts are excluded: the first train invocation correctly rejected
mixture plus replay before training, and the first eval invocation lacked the
AgentV runtime and did not complete the required bundle.

## Matched OOD result

| Metric | E520 / E519 control | E523 / E522 visible inventory | Delta |
| --- | ---: | ---: | ---: |
| Meaningful | 0.0000 | 0.0000 | 0.0000 |
| Placeholder fidelity | 0.4083 | 0.8667 | +0.4583 |
| Structure | 0.2250 | 0.1955 | -0.0295 |
| Component recall | 0.2083 | 0.2708 | +0.0625 |
| Reward | 0.7445 | 0.2093 | -0.5353 |
| AST node F1 | 0.2833 | 0.3437 | +0.0603 |
| AST edge F1 | 0.0625 | 0.1007 | +0.0382 |
| Strict binding-aware meaning | 0.0000 | 0.0000 | 0.0000 |
| AgentV | 0/1 | 0/1 | — |

The n=4 diagnostic preserves syntax and dramatically improves slot fidelity,
but three outputs remain trivial and every output has placeholder-role mismatch
or spam. The fidelity gain therefore does not compose into meaningful
hierarchy.

## Decision

Retain visible prompt inventory as a real positive data lever. Reject the E522
checkpoint: structure remains below the OOD floor, meaningful and strict
meaning remain zero, and AgentV is red. The next matched intervention should
pair visible inventory with component-hierarchy supervision or data, not
increase slot loss. Machine-readable evidence is in
[the E522 JSON](iter-e522-visible-slot-continuation-20260719.json).
