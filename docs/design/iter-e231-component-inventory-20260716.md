# E231 — request-level component inventory

Date: 2026-07-16  
Status: completed; checkpoint rejected; not promotable or ship

## Hypothesis and boundary

E231 tests whether a request-level component inventory can break the generic
`Stack` → `TextContent` semantic prior left by E230. The target is generated
from lexer `TokenKind.COMPONENT` membership in each gold document. A pooled
prompt representation predicts the multi-label inventory with balanced positive
and negative BCE. During compiler decode, those logits can only reorder component
tokens already admitted by the compiler completion forest. They cannot add,
remove, or repair legal tokens, and no prompt, output, component arrangement, or
string-literal case is encoded in the implementation.

Immediately before training, `origin/main` was fetched twice. The isolated
branch rebased cleanly onto `0ecdb36`, had zero remote commits ahead, and had no
uncommitted files. Focused verification after the rebase passed 74 tests (3
deselected) plus Ruff.

## Train

The run used the published `e230_diverse_judged_roots_v2` corpus: 126
independently judged roots, CPU, 32 steps, batch 4, learning rate 0.0003, seed 0,
frozen local SmolLM2-135M context, lexer output, schema and honest slot context,
no DESIGN context, compiler candidate CE plus margin 1.0, inventory loss/decode
weights 1.0, capacity-aware sampling, and no checkpoint sync. It consumed 18,490
prompt and 7,052 target tokens in 165.62 s. The 128 draws covered 81 unique rows,
including 30 RICO and 25 human-curated draws. Training trace:
`8549c53f5a3da36bf6d74e8e8696664e`.

The auxiliary objective learned its training target:

| Metric | Step 1 | Step 32 |
| --- | ---: | ---: |
| Inventory BCE | 1.2650 | 0.5322 |
| Gold-count top-k recall | 0.0000 | 0.9167 |
| Positive minus negative logit margin | 0.3726 | 2.7841 |

Final total loss was 19.9879, bound-component alignment loss was 3.3775, and
compiler-margin violation rate was 0.5000. Checkpoint SHA-256:
`136aa0043df98d52d29b1f9cbcce5f0a8f5d8b5dbd5cbf883a1a78f2b6d475de`.

## Honest evaluation

Strict compiler-tree evaluation used all five suites, the unchanged honest slot
contract, no unconstrained fallback, AgentEvals JSONL, and the pinned AgentV SDK.
Evaluation trace: `aa0eff55cbb1d608f1f4826fa9435237`.

| Suite | n | syntax | meaningful | structure | component recall | fidelity | contract precision | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.4636 | 0.2500 | 0.1944 | 0.6667 | 0.4910 |
| held_out | 5 | 1.0000 | 0.0000 | 0.3302 | 0.1567 | 0.1133 | 0.6000 | 0.4234 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.4681 | 0.4583 | 0.4583 | 0.7500 | 0.6242 |
| ood | 4 | 1.0000 | 0.0000 | 0.3469 | 0.1458 | 0.2083 | 0.7500 | 0.5493 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1628 | 0.4444 | 0.1250 | 1.0000 | 0.6865 |

Six thresholds fail across four suite cases. AgentV passes 1/5 with mean score
0.6 and zero execution errors. Syntax remains deterministic at 1.0, adversarial
component recall improves over E230, but smoke/held-out/OOD fidelity and reward
regress while the generic component pattern remains.

## Bias-off causal ablation

A second full evaluation set the inventory decode weight to zero while preserving
the same checkpoint and all other policy. Every aggregate metric and every
component choice was identical. Five records changed only placeholder or style
surface choices. AgentV remained 1/5. Ablation trace:
`cf5f86af79d7d217a0c73216eba8ee44`.

The pooled inventory head therefore learns the train target but does not provide
a useful held-out component-ranking intervention. Keep the generalized loss,
telemetry, compiler-legal bias boundary, and eval override for future controlled
experiments; reject this checkpoint and do not infer that more steps will fix the
semantic hierarchy.

Machine-readable evidence:
[iter-e231-component-inventory-20260716.json](iter-e231-component-inventory-20260716.json).
