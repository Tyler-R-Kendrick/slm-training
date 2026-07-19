# E525 — visible component-contract continuation

E525 tests whether E524’s prompt-visible component counts restore hierarchy
while preserving E522’s slot-fidelity gain. The parent, 50% E357 replay,
5k-token budget, honest context authority, objective weights, seed, and
evaluator are fixed. E524 has the same 244 IDs and OpenUI targets as E521, so
the only primary-corpus difference is the component contract in each prompt.

## Train, persistence, and checkpoint

The clean-source CPU HF-context run completes 99 steps / 5,059 target tokens in
76.72 seconds under `max_wall_minutes=3`. The first invocation is invalid
setup: it omitted the non-default choice tokenizer and failed parent tensor
shape checks before any training step.

Automatic bucket sync then hit the sandbox network boundary after training.
Canonical rescue sync uploaded and verified the complete nine-file bundle, and
an independent HF listing confirmed it at
`hf://buckets/TKendrick/OpenUI/checkpoints/e525-e396-e524-replay050-slotrole1-honest-context-r2-5k/`.
The serving SHA is `dbd11811…e55e4b9` and the full-state SHA is
`dd588c80…ea9ad63`. The rescue workflow now writes its verified report back to
`train_summary.json` and `checkpoint_bucket.json`.

## Matched OOD result

| Metric | E523 / E522 slot contract | E526 / E525 component contract | Delta |
| --- | ---: | ---: | ---: |
| Meaningful | 0.0000 | 0.0000 | 0.0000 |
| Placeholder fidelity | 0.8667 | 0.4667 | -0.4000 |
| Structure | 0.1955 | 0.1452 | -0.0503 |
| Component recall | 0.2708 | 0.4167 | +0.1458 |
| Reward | 0.2093 | 0.1668 | -0.0425 |
| AST node F1 | 0.3437 | 0.3041 | -0.0396 |
| AST edge F1 | 0.1007 | 0.0774 | -0.0233 |
| Strict binding-aware meaning | 0.0000 | 0.0000 | 0.0000 |
| AgentV | 0/1 | 0/1 | — |

The n=4 diagnostic retains syntax and increases component type recall, showing
that the visible count signal is learned. It does not compose into reference
hierarchy: two outputs are trivial, three miss required prompt components, and
meaningful/strict meaning remain zero.

## Decision

Reject E525. Exact component counts are conditional-contract evidence and
improve recall, but they trade away slot fidelity and hierarchy. Do not
increase count prompting or repeat the previously negative hidden topology
losses. The next intervention should address prompt-to-reference-graph
construction without exposing exact gold counts. Machine-readable evidence is
in [the E525 JSON](iter-e525-visible-component-continuation-20260719.json).
