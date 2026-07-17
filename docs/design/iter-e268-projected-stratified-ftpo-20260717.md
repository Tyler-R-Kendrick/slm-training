# E268 — conflict-projected decision-kind safe set FTPO

Date: 2026-07-17
Status: **completed; PCGrad direction unsafe; parent restored**

E268 tests the generalized follow-up to E266/E267: construct one `ftpo_set`
gradient for every grammar/AST `decision_kind`, deterministically project each
gradient against negatively aligned peer gradients, average the projected
gradients, and retain the unchanged optimizer-consistent backtracking plus
held-out per-kind Pareto guard. The implementation derives categories from the
committed decision events; it contains no prompt, token, or string-literal
case rules.

Immediately before the real training invocation, the branch fetched and
rebased latest `origin/main`, was clean, and proved `0 behind / 1 ahead` at
harness commit `e927fae`. The first invocation resolved the editable package
from the shared checkout and failed before training with an unsupported-wrapper
argument. Its exception-only JSON was discarded; the clean gate was repeated,
and the completed run explicitly selected this worktree with `PYTHONPATH=src`.
Training trace: `82c14b2da0f2f294acb93388fe72fe08`. Evaluation trace:
`6ee4dbb53b55e22034577e806f753a3c`.

## Result

The 30 matched steps computed 420 task gradients. Of 5,460 ordered task pairs,
2,220 had negative dot products and were projected. Nevertheless, every one of
the 30 proposals failed all five tested scales (`1, 1/2, 1/4, 1/8, 1/16`):
0 steps were accepted and the parent was restored.

Several blockers occurred on all 150 trials, including:

| Decision kind | Metric | Regressing trials |
| --- | --- | ---: |
| `bind_declaration_root` | good probability mass | 150 |
| `bind_reference_bound_children` | loss | 150 |
| `bind_reference_root_children` | good probability mass | 150 |
| `bind_reference_root_children` | mean margin | 150 |
| `component_bound` | good probability mass | 150 |
| `component_root` | good probability mass | 150 |
| `grammar_comma` | loss | 150 |
| `grammar_comma` | bad probability mass | 150 |
| `grammar_comma` | good probability mass | 150 |
| `grammar_comma` | mean margin | 150 |
| `lit` | good probability mass | 150 |
| `lit` | mean margin | 150 |

All held-out deltas are exactly zero after restoration. Model tensors are
bit-identical to E228, and the serialized checkpoint SHA is
`518d4736571df2f3842ffd338801cfcc4a855d50358c87bd7563facb191935ba`.

## Performance and full evaluation

The local stage took 2,338.56 seconds (38m59s) for 150 validation batches and
5,850 logical held-out event checks. This is 25.9x E267's 90.27-second stage.
The cost comes from retaining the full 200-event graph while computing 14 task
gradients per step; this implementation is not a practical replacement even
apart from its rejected quality result.

The restored checkpoint exactly reproduces E266/E267/current-parent quality:

| Suite | n | Syntax | Meaningful | Fidelity | Structure | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.7222 | 0.5100 | 0.8777 |
| held_out | 5 | 1.0000 | 0 | 0.5600 | 0.3943 | 0.8290 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.8333 | 0.4654 | 0.9110 |
| ood | 4 | 1.0000 | 0 | 0.5167 | 0.4081 | 0.8160 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.2500 | 0.2355 | 0.7360 |

Five ship thresholds fail. AgentV passes 2/5 with zero execution errors.
Syntax remains deterministic at 1.0 with zero constrained fallbacks and decode
timeouts.

## Decision

Reject and do not sync or promote E268. Pairwise PCGrad conflict removal is not
a common-descent certificate: sequential projections can reintroduce conflicts,
and non-conflicting train-loss gradients need not preserve every held-out guard
metric at a finite optimizer step. Do not tune duration, scalar learning rate,
or individual decision kinds.

The next generalized hypothesis is a deterministic minimum-norm convex
combination of per-kind gradients with an explicit common-descent certificate
before optimizer backtracking. Benchmark one projected step before a matched
30-step run, and retain the strict grammar/AST guard.

Machine-readable evidence:
[`quality-matrix-v10-e268-results.json`](quality-matrix-v10-e268-results.json).
