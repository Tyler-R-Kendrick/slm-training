# E267 — decision-kind block-coordinate safe set FTPO

Date: 2026-07-17
Status: **completed; every category-block direction unsafe; parent restored**

E267 tests whether E266 failed because single-event gradients were too noisy.
It groups all committed E261 train events sharing one grammar/AST
`decision_kind`, averages their `ftpo_set` losses into one block proposal, and
applies the unchanged decision-kind-stratified Pareto guard. Fourteen decision
kinds contain 1–23 train events; the 30-step schedule cycles those blocks and
tests scales `1, 1/2, 1/4, 1/8, 1/16` from the last accepted model and Adam
state.

Immediately before training, the branch fetched and rebased latest
`origin/main`, was clean, and proved `0 behind / 1 ahead` at harness commit
`28ccafcda31ba3a070cbdf85b7c27b91775a7c8b`. Training trace:
`542865f6abb6538fe8cc5a9743091fba`. Evaluation trace:
`f4d437af70bcbd5330cc95161fe0e27f`.

## Result

All 30 category-block proposals and all 150 scales were rejected. Held-out
metrics have exactly zero delta. The serialized checkpoint SHA is
`518d4736571df2f3842ffd338801cfcc4a855d50358c87bd7563facb191935ba`;
all 374 tensors and the model config are bit-identical to E228.

Frequent blockers include:

| Decision kind | Metric | Regressing trials |
| --- | --- | ---: |
| `grammar_comma` | good probability mass | 120 |
| `component_root` | good probability mass | 115 |
| `component_root` | mean margin | 110 |
| `grammar_comma` | loss | 106 |
| `grammar_comma` | mean margin | 106 |
| `bind_reference_bound_children` | bad probability mass | 89 |
| `lit` | good probability mass | 88 |
| `component_root` | loss | 86 |
| `lit` | mean margin | 85 |

Averaging within a semantic category therefore does not reveal a safe FTPO
direction. The conflict is not merely per-event gradient noise.

## Performance and full evaluation

The batched local stage took 90.27 seconds for 150 validation batches, only
10.49 seconds slower than E266 despite training on whole category blocks. The
same batching/caching path remains practical.

The full evaluation exactly matches E266 and its current-code E248 parent
control:

| Suite | n | Syntax | Meaningful | Fidelity | Structure | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.7222 | 0.5100 | 0.8777 |
| held_out | 5 | 1.0000 | 0 | 0.5600 | 0.3943 | 0.8290 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.8333 | 0.4654 | 0.9110 |
| ood | 4 | 1.0000 | 0 | 0.5167 | 0.4081 | 0.8160 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.2500 | 0.2355 | 0.7360 |

Five ship thresholds fail and AgentV passes 2/5 with zero execution errors.
Syntax remains deterministic at 1.0 with zero fallback and timeout counts.

## Decision

Reject and do not sync or promote E267. Single-event and category-averaged FTPO
directions are both outside the strict grammar/AST Pareto cone at every tested
scale. Do not spend another run on duration, a smaller scalar learning rate, or
case-specific exceptions. The next generalized hypothesis should combine
per-kind gradients with conflict projection or a minimum-norm constrained
solver, so the proposal direction is constructed to preserve all guarded
categories before backtracking tests its magnitude.

Machine-readable evidence:
[`quality-matrix-v10-e267-results.json`](quality-matrix-v10-e267-results.json).
The matched parent evidence remains
[`quality-matrix-v10-e266-current-parent-control-results.json`](quality-matrix-v10-e266-current-parent-control-results.json).
