# E222 — capacity-aware exposure diagnostic

Status: **training and strict five-suite evaluation completed; sampler mechanism
confirmed; semantic non-regression falsified; ship gates failed; no checkpoint
promoted**.

E221 showed that task-balanced sampling with replacement exposed only 29.68
effective records in 128 draws and repeated one renderer row 18 times. E222
tested a generalized capacity-aware policy: preserve hierarchical task/family
weights, but draw rows without replacement inside each bounded sampler window.
The implementation is deterministic, includes the policy in resume identity,
and persists pending batches for bit-exact resume.

Immediately before launch, latest `main` was fetched and the clean branch was
confirmed zero commits behind. A five-candidate matrix selected the matched
policy comparison. The executed member changed only
`mixture_sampling_policy=capacity_aware` from E221: canonical 480-record E218
data, canonical task/family weights, frozen local SmolLM2-135M context, lexer
output, stratified compiler alignment, schema and slot context, 32 CPU steps,
batch 4, learning rate 0.0003, seed 0, no DESIGN context, no checkpoint sync,
and strict tree evaluation with unconstrained fallback disabled.

Training completed with last loss 11.7409 over 23,609 prompt and 6,534 target
tokens in 144.01 s. The local checkpoint SHA-256 is
`960e13f1…3f348c5`; trace ID is `08e3e21930a50c48fedcc709206a456f`.

## Exposure result

| Run | draws | unique | effective | effective ratio | max repeat |
| --- | ---: | ---: | ---: | ---: | ---: |
| E221 task-balanced replacement | 128 | 80 | 29.68 | 0.2319 | 18 |
| E222 capacity-aware | 128 | 102 | 83.59 | 0.6531 | 4 |

The sampler mechanism is confirmed: effective exposure increased 2.82× and the
maximum repeat fell from 18 to four. Family draws were corruption repair 57,
edit trajectory 43, ProgramSpec generation 20, renderer visual 4, and web
distilled 4. Task-group draws were repair/completion/inpaint 57, generation 39,
and patch/edit 32. Capacity bounds therefore fixed row repetition, but stochastic
weighted selection did not tightly preserve equal task quotas in this 128-draw
sample. The next sampler hypothesis should combine capacity-aware row selection
with deterministic largest-remainder task/family quotas, rather than tuning a
specific family weight.

## Strict evaluation

| Suite | n | syntax | meaningful parse | structure | component recall | fidelity | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.6667 | 0.0000 | 0.2661 | 0.0833 | 0.0000 | 0.2123 |
| held_out | 5 | 0.6000 | 0.0000 | 0.2796 | 0.1067 | 0.0000 | 0.2548 |
| adversarial | 4 | 0.7500 | 0.5000 | 0.3845 | 0.4583 | 0.0000 | 0.4778 |
| ood | 4 | 0.7500 | 0.0000 | 0.3719 | 0.0625 | 0.0000 | 0.1593 |
| rico_held | 3 | 0.0000 | 0.0000 | 0.1501 | 0.4444 | 0.0000 | 0.0000 |

Ten ship checks failed; AgentV passed 1/5 suite records with no execution
errors. Every suite recorded zero compiler fallback count and the persisted
policy has `allow_unconstrained_fallback=false`. Compared with E221, smoke
meaningful parse regressed from 0.3333 to zero and syntax from 1.0 to 0.6667,
although adversarial meaningful parse improved from 0.25 to 0.50. The combined
hypothesis is therefore falsified: improved exposure alone did not preserve
quality at this short training budget. The capacity-aware mechanism remains
useful infrastructure, but this checkpoint is diagnostic only and is neither
synced, promoted, nor shippable.
