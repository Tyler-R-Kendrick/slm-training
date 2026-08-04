# Autotrain c1830: capacity-aware tail fixture positive

**Verdict:** queue exact fresh-seed confirmation. Tail-weighted scaffold
supervision improves structure, binder F1, fidelity, and reward while trimming
tokens and forwards at matched exposure and parameter count. The 6.1% p50
increase remains inside the registered screening budget.

| Arm | Params | Effective / draws | Unique | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | Tokens | Forwards | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capacity-aware margin control | 1,608,962 | 27.59 / 40 | 32 | .35057 | .3333 | .25 | .7222 | .6111 | .8443 | 104 | 23 | 6047.96 |
| + tail supervision | 1,608,962 | 27.59 / 40 | 32 | .40333 | .3333 | .25 | .8222 | .7222 | .8777 | 99 | 22 | 6417.56 |

The treatment changes only `ltr_tail_loss_weight` from 0 to 1. Capacity-aware
sampling, all-family legal margin, seed, steps, corpus, model shape, and eval
policy are identical. Structure rises `.05277`, binder F1 `.10`, fidelity
`.1111`, and reward `.0333`; MPR and recall hold. Tokens fall 4.8% and forwards
4.3%. Compiler time rises 7.2% and p50 6.1%, so confirmation must reproduce both
quality and the bounded-cost disposition rather than relying on this small
sample.

This remains fixture evidence: `n=3<20`, MPR and recall miss their gates, AST
and canonical equality are zero, and held-out, adversarial, OOD, and full Rico
were not run. Both arms use seed 101830, 20 CPU scratch steps, batch size 2, and
1,608,962 trainable parameters. Candidate SHA `a4e8d972...cafcd1`; control SHA
`420d3e8a...58b0b`. Checkpoints are local no-sync artifacts and are never
reusable, promotable, syncable, or shippable. Lean is
`not_applicable:screening`; formal preflight stays locked until confirmation.

Machine evidence:
[`autotrain-cycle-1830-capacity-aware-tail-positive.json`](autotrain-cycle-1830-capacity-aware-tail-positive.json).
