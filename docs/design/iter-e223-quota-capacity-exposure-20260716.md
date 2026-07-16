# E223 — quota-capacity exposure diagnostic

Status: **training and strict evaluation completed; allocation mechanism
confirmed; quality-recovery hypothesis falsified; 12 ship gates failed; no
checkpoint promoted**.

E222 fixed record repetition but stochastic capacity sampling shifted the three
usable task groups to 57/39/32 draws and regressed smoke syntax and meaningful
parse. E223 added a generalized largest-remainder quota layer with capacity
redistribution at task and family levels, followed by random row selection
without replacement. No family names or example literals are special-cased.

Latest `main` was fetched immediately before the run; the clean branch was zero
commits behind. A five-candidate matrix selected the exact E222 recipe with only
`mixture_sampling_policy=quota_capacity_aware` changed: canonical E218 train and
`remediated` eval data, 32 CPU steps, batch 4, learning rate 0.0003, seed 0,
frozen local HF context, lexer output, stratified compiler alignment, schema and
slot context, no DESIGN context, no checkpoint sync, tree decode, and
unconstrained fallback disabled.

Training completed with loss 11.9060 over 22,924 prompt and 6,401 target tokens
in 146.04 s. Checkpoint SHA-256 is `2db1e797…28a5ab87`; trace ID is
`d564a679da92a72dc3ce3519a5de66a6`. The checkpoint remains local.

## Allocation and exposure

| Run | task draws | unique | effective | ratio | max repeat |
| --- | --- | ---: | ---: | ---: | ---: |
| E222 stochastic capacity | 57 / 39 / 32 | 102 | 83.59 | 0.6531 | 4 |
| E223 quota capacity | 44 / 44 / 40 | 103 | 81.11 | 0.6337 | 4 |

Task order above is the sorted observed allocation, with E223 explicitly
generation 44, patch/edit 44, and repair/completion/inpaint 40. The quota
mechanism is confirmed and retains high exposure. Family draws were edit
trajectory 52, corruption repair 40, ProgramSpec generation 28, renderer visual
4, and web distilled 4.

## Strict evaluation

| Suite | n | syntax | meaningful parse | structure | component recall | fidelity | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.0000 | 0.3094 | 0.0000 | 0.0000 | 0.0000 |
| held_out | 5 | 1.0000 | 0.0000 | 0.2514 | 0.0000 | 0.0000 | 0.0000 |
| adversarial | 4 | 1.0000 | 0.0000 | 0.2905 | 0.0000 | 0.0000 | 0.0000 |
| ood | 4 | 1.0000 | 0.0000 | 0.2369 | 0.0000 | 0.0000 | 0.0000 |
| rico_held | 3 | 1.0000 | 0.0000 | 0.0901 | 0.0000 | 0.0000 | 0.0000 |

All suites had zero compiler fallbacks and zero constrained-fallback rate, so
the deterministic lexical layer worked. Meaningful parse, component recall,
fidelity, and reward were zero throughout. Twelve gates failed and AgentV
passed 0/5 with no execution errors. Quota balancing is therefore not the next
quality lever: it repairs sampler invariants but does not supply the learned
semantic decisions needed inside valid programs. Further work should inspect
semantic alignment targets and model-selected AST roles under the now-stable
sampler, not add another tactical sampling policy.
