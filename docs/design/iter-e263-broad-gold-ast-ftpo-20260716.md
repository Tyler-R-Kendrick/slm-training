# E263 — broad gold-AST set FTPO

Date: 2026-07-16
Status: **completed; hypothesis falsified; 10 ship thresholds failed; checkpoint rejected**

E263 isolates the data-support hypothesis raised by E252. It repeats the same
30 CPU `ftpo_set` updates at learning rate `5e-5` from the unchanged E228
parent, but replaces the narrow 16-event E256 corpus with the committed E261
grammar/AST-aligned corpus: 200 train and 39 held-out events across 53/11 prompt
groups, 14 decision kinds, and 108 set-valued comparisons. No historical step
output was used as training data.

Immediately before training, the branch fetched and rebased latest
`origin/main`, was clean, and proved `0 behind / 1 ahead` (the ahead commit was
the tested broad-FTPO matrix registration). The run originally emitted E262 and
`qx_e262_broad_gold_ast_ftpo_set`; it was renumbered E263 during merge conflict
resolution because concurrent B1 work registered E262 on `main`. Parent SHA is
`7a9be4a665e216d7f7e73883ad74ad972bbf30846896d0c29188d6482f5b093a`;
the rejected E263 checkpoint SHA is
`3f6a2eb2a6b326fc394add5b4588d81f4bbf648d0a39c68cc53e5046f760831b`.

## Exact-state result

The broad held-out set did not improve under the matched update budget:

| Metric | Parent | E263 | Delta |
| --- | ---: | ---: | ---: |
| FTPO loss | 2.7660 | 3.0144 | +0.2484 |
| Chosen win | 0.6325 | 0.6410 | +0.0085 |
| Margin win | 0.4872 | 0.5128 | +0.0256 |
| Mean good-minus-bad margin | 1.0035 | 0.8035 | -0.2001 |
| Good probability mass | 0.041869 | 0.056764 | +0.014896 |
| Bad probability mass | 0.039225 | 0.082156 | +0.042931 |

Training sampled 11 decision kinds, but the larger rise in bad-token mass and
worse held-out loss show that unguarded set FTPO did not generalize across the
held-out exact states. More data alone does not validate the objective.

## Full evaluation

| Suite | n | Syntax | Meaningful | Fidelity | Structure | Reward | E248 structure delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.5278 | 0.1742 | 0.7653 | -0.2900 |
| held_out | 5 | 1.0000 | 0 | 0.2800 | 0.1088 | 0.6910 | -0.2281 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.5417 | 0.1927 | 0.7695 | -0.2817 |
| ood | 4 | 1.0000 | 0 | 0.2583 | 0.1469 | 0.6845 | -0.2281 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1250 | 0.0727 | 0.6445 | -0.0901 |

Syntax stayed 1.0 with zero compiler fallbacks, unconstrained fallbacks, and
decode timeouts. This is the deterministic compiler-tree layer doing its job;
it is not learned structural quality. Placeholder fidelity and
meaningful-program rates exactly match the E248 parent, so broad support avoids
E252's fidelity collapse. Structural similarity nevertheless regresses on all
five suites, and reward falls by 0.042 on every suite. Ten unchanged ship
thresholds fail. AgentV passes 0/5 with zero execution errors.

## Harness interruption and recovery

The sole training stage completed before publication failed because an isolated
Git worktree did not contain root `node_modules`, so Node could not resolve the
pinned `@agentv/core` package. The matrix correctly retained the checkpoint,
five domain eval files, local preference summary, and traces. The harness now
resolves the SDK from the Git common checkout and supports recipe-validated
`--resume`; E263 reused the completed stage only after matching objective,
steps, split counts, balancing/tether settings, parent identity, and checkpoint.
No second training stage ran.

Training trace: `3669a8ab4509b7d807f343c72722e51a`. Evaluation trace:
`a1f3fdb00a34f984c570740155a2bfb6`.

## Decision

Reject E263 and do not promote or continue its checkpoint. Broad judged
grammar/AST support is necessary and remains the canonical corpus, but a fixed
30-step unguarded set-FTPO objective is not sufficient: it worsens held-out
preference loss and shifts legal decisions away from semantic structure while
the deterministic layer keeps syntax green. The next hypothesis should gate
updates against held-out exact-state regression and preserve the E228 semantic
control, rather than adding more syntax supervision or simply extending
training duration.

Machine-readable evidence:
[`quality-matrix-v10-e263-results.json`](quality-matrix-v10-e263-results.json).
This is a rejected local process checkpoint, so it was not synced to the HF
bucket and was not promoted.
