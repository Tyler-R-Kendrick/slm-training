# E252 — verifier-backed set FTPO

Date: 2026-07-16
Status: **completed; hypothesis falsified; 13 ship thresholds failed; checkpoint rejected**

E252 applied 30 CPU `ftpo_set` updates at learning rate `5e-5` to the unchanged
E228 parent. It consumed the committed
`e256_counterfactual_semantic_v1` corpus: 14 train and two held-out events,
eight set-valued comparisons, and only independently judged counterfactual
evidence. The run used seed 0, frozen local HF context, strict compiler-tree
decoding, all five committed remediated suites, and unchanged ship gates.

Immediately before training, the branch fetched and rebased latest
`origin/main`, was clean, and proved `0 behind / 0 ahead`. Parent SHA is
`7a9be4a665e216d7f7e73883ad74ad972bbf30846896d0c29188d6482f5b093a`;
the rejected E252 checkpoint SHA is
`c01aebc28d8f873750378c364d0ec244795c171ae9ad8d8e560cc44e46088946`.

## Exact-state result

The two held-out events show mixed local movement:

| Metric | Parent | E252 | Delta |
| --- | ---: | ---: | ---: |
| FTPO loss | 5.3048 | 2.9643 | -2.3405 |
| Chosen win | 0.1667 | 0.3333 | +0.1667 |
| Margin win | 0 | 0.3333 | +0.3333 |
| Mean good-minus-bad margin | -3.2891 | 1.0663 | +4.3553 |
| Good probability mass | 0.001359 | 0.000899 | -0.000460 |
| Bad probability mass | 0.039896 | 0.009845 | -0.030051 |

The component-root event improved strongly, but the bind-declaration event
remained wrong and good-token mass decreased overall. Two held-out events from
one group are enough to test the predefined prerequisite, not enough to claim a
broad semantic preference effect.

## Full evaluation

| Suite | n | Syntax | Meaningful | Fidelity | Structure | Reward | E248 structure delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0 | 0.1353 | 0.6070 | -0.3289 |
| held_out | 5 | 1.0000 | 0.2000 | 0 | 0.1239 | 0.6070 | -0.2129 |
| adversarial | 4 | 1.0000 | 0.2500 | 0 | 0.0978 | 0.6070 | -0.3765 |
| ood | 4 | 1.0000 | 0.2500 | 0 | 0.1906 | 0.6070 | -0.1844 |
| rico_held | 3 | 1.0000 | 0 | 0 | 0.0368 | 0.6070 | -0.1259 |

Syntax remained 1.0 with zero compiler fallbacks and zero decode timeouts. That
is the deterministic grammar/compiler layer working as intended, not evidence
that the model learned structure. Placeholder fidelity collapsed to zero on
every suite; structure and reward regressed on every suite versus E248. Thirteen
unchanged ship thresholds failed, and AgentV passed 0/5 with zero execution
errors.

## Decision

Reject E252 and do not promote or continue it. The semantic judge correctly
filtered individual candidate completions, but the admitted data covers only
root binding and root component decisions. Repeating those narrow decisions for
30 updates overfits local rankings and destroys broader placeholder/content
behavior. The next data hypothesis must increase decision-depth and prompt-group
coverage before any E253/E254 training; a tether or balancing knob cannot repair
missing semantic support.

Training trace: `c5f845b567bcf9c4eb6d82b2c6f1cce3`. Evaluation trace:
`1688e3129ba108fd3c059cf618d000b7`. AgentEvals and pinned AgentV artifacts are
under `outputs/autoresearch/e252-ftpo-set/runs/qx_e252_local_ftpo_set/agentv/`.

Machine-readable evidence:
[`quality-matrix-v10-e252-results.json`](quality-matrix-v10-e252-results.json).
This is a rejected local process checkpoint, so it was not synced to the HF
bucket and was not promoted.
