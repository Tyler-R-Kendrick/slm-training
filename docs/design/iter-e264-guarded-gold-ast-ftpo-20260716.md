# E264 — held-out guarded gold-AST set FTPO

Date: 2026-07-16
Status: **completed; guard admitted no update; checkpoint is parent-equivalent and rejected**

E264 tests whether the E263 set-FTPO objective can be made safe by selecting
only checkpoints that do not regress held-out exact-state behavior. It repeats
E263's 30 CPU `ftpo_set` steps at learning rate `5e-5` over the committed E261
corpus (200 train and 39 held-out events), but validates every five steps. A
candidate is eligible only when it weakly improves all four guard metrics:
held-out loss and bad-token probability mass must not increase, while
good-token probability mass and mean good-minus-bad margin must not decrease.

Immediately before training, the branch fetched and rebased latest
`origin/main`, was clean, and proved `0 behind / 1 ahead`; the ahead commit was
the tested guard harness. Training trace:
`fbdaa4deb64c2fa4e05589cf67596e9f`. Evaluation trace:
`816914ca9963950b41da375d3362d830`.

## Guard result

No trained checkpoint satisfied the Pareto guard:

| Step | Eligible | Loss | Bad mass | Good mass | Mean margin |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 0 | yes | 2.7660 | 0.039225 | 0.041869 | 1.0035 |
| 5 | no | 3.0379 | 0.047729 | 0.044631 | 1.1830 |
| 10 | no | 2.9255 | 0.061491 | 0.048399 | 0.9199 |
| 15 | no | 2.9258 | 0.066424 | 0.052240 | 0.8194 |
| 20 | no | 2.8572 | 0.062408 | 0.058178 | 0.9755 |
| 25 | no | 2.9159 | 0.068721 | 0.055183 | 0.8943 |
| 30 | no | 3.0144 | 0.082156 | 0.056764 | 0.8035 |

The harness therefore restored step 0. Before/after held-out metrics have zero
delta. Direct artifact verification confirms all 374 model tensors are
bit-identical to E228 (`max_abs_delta=0`), with identical model config and
tokenizer sidecars. The E264 file SHA
`518d4736571df2f3842ffd338801cfcc4a855d50358c87bd7563facb191935ba`
differs only because the parent-equivalent payload was serialized again; it is
not a new learned model.

## Current-code parent control

Because the parent-equivalent E264 evaluation scored differently from the
historical E248 record, the unchanged E228 checkpoint was evaluated again with
the same current code, suites, policy, and eight decode steps. The current
E248 control reproduced E264 exactly for every suite metric and all five gate
failures. Its evaluation trace is `2654051b3d257c73a325eb0678d736a9`.

| Suite | n | Syntax | Meaningful | Fidelity | Structure | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.7222 | 0.5100 | 0.8777 |
| held_out | 5 | 1.0000 | 0 | 0.5600 | 0.4076 | 0.8314 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.8333 | 0.4654 | 0.9110 |
| ood | 4 | 1.0000 | 0 | 0.5167 | 0.4081 | 0.8160 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.2500 | 0.2355 | 0.7360 |

Syntax is 1.0 with zero fallback and timeout counts in both runs. That remains
evidence for the deterministic compiler-tree path, not learned structure.
Relative to historical E248, current-code fidelity and reward are higher while
meaningful-program rates are unchanged; current component-recall gates also
produce five failures. Those changes belong to intervening evaluator/decoder
code and must not be attributed to E264 training.

## Decision

Keep the generalized held-out guard: it correctly prevented the known E263
regression and made unsafe optimization a no-op. Reject the E264 artifact and
do not sync or promote it because no trained step was admitted and the restored
checkpoint is bit-identical to E228. The next optimization hypothesis must
change the objective or update geometry enough to improve all guarded
exact-state metrics; extending the same unguarded trajectory is already
falsified.

Machine-readable evidence:
[`quality-matrix-v10-e264-results.json`](quality-matrix-v10-e264-results.json)
and the matched
[`quality-matrix-v10-e264-current-parent-control-results.json`](quality-matrix-v10-e264-current-parent-control-results.json).
