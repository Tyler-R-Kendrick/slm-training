# E531 — visible semantic-role continuation

E531 tests whether E530's prompt-visible semantic namespaces and
schema-compatible owning-type candidates improve hierarchy without exposing
exact output counts or the gold reference graph.

The train holds E528's E396 parent, exact E357 replay, 50% replay fraction,
5,000-token budget, objective weights, honest authority, and choice tokenizer
fixed. Only the membership-identical primary data projection changes from E527
type inventories to E530 semantic roles.

The clean CPU HF-context run completed 99 steps / 5,059 target tokens in 99.72
seconds under `max_wall_minutes=3`. The automatic sync's unnecessary
bucket-create preflight was rejected by the CLI OAuth session, so the canonical
rescue sync targeted the already-existing bucket directly. It reconciled
`train_summary.json`, verified the upload by an empty resync plan, and an
independent listing confirmed all nine files.

| Metric | E529 / E528 type contract | E532 / E531 semantic roles | Delta |
| --- | ---: | ---: | ---: |
| Syntax parse rate | 1.0000 | 1.0000 | 0.0000 |
| Meaningful program rate v1 | 0.2500 | 0.0000 | -0.2500 |
| Placeholder fidelity | 0.5500 | 0.4667 | -0.0833 |
| Structural similarity | 0.1136 | 0.1431 | +0.0296 |
| Component type recall | 0.3542 | 0.2917 | -0.0625 |
| Reward | 0.5778 | 0.3685 | -0.2093 |
| AST node F1 | 0.2270 | 0.2543 | +0.0273 |
| AST edge F1 | 0.0801 | 0.0455 | -0.0347 |
| Strict binding-aware meaning | 0.0000 | 0.0000 | 0.0000 |
| AgentV | 0 / 1 | 0 / 1 | unchanged |

The role projection weakly improves structure and AST node overlap, but does
not cause correct reference-graph construction. Three of four outputs miss
required prompt components/placeholders; two have low component recall, and
strict meaning remains zero.

Reject E531 for promotion. Keep E530/E531 as evidence that prompt grouping
alone is insufficient. The next lever should train or decode explicit
reference edges from visible contracts without exposing the gold graph or
weakening ship gates. Machine-readable evidence is in
[the E531 JSON](iter-e531-visible-semantic-roles-continuation-20260719.json).
