# E235 — binder-instance component plan

Date: 2026-07-16
Status: completed; checkpoint rejected; not promotable or ship

E235 replaces partial-prefix parent recovery with a prompt-conditioned plan over
the compiler's grammar-native binder slots. Each `component_bound` decision is
supervised by cross-entropy over only the component candidates admitted by the
compiler forest for that binder. Decode bias is scoped to the active binder and
cannot change grammar legality. The implementation contains no fixture-specific
or component-name string cases.

Immediately before training, the isolated branch fetched and rebased onto
`origin/main` at `1a7e12e`, was zero commits behind, and was clean. Ruff,
compile checks, and 79 focused tests passed against the isolated source.

The matched E234 recipe used the published 126-row
`e230_diverse_judged_roots_v2` corpus, CPU, 32 steps, batch 4, learning rate
0.0003, seed 0, frozen local SmolLM2-135M, lexer output, exhaustive compiler
CE/margin 1.0, role/count loss and decode weight 1.0, binder-plan loss and
decode weight 1.0, all edge objectives and edge decode weights 0, schema and
train-only slot context, no DESIGN context, capacity-aware sampling, and no
checkpoint sync. The 128 draws covered 81 unique rows, including 30 RICO and
25 human-curated draws. Training took 160.56 s; trace:
`d76289e4b6ed374ceef711488142ed45`; checkpoint SHA:
`83adbccd0f05cb4afcf9debd446e15270fd1ee62d3ed4c153e52646950073ca8`.

Binder-plan accuracy rose 0 → 0.4000 and loss fell 3.8573 → 2.2583. The final
sampled batch supervised all 30 bound-component rows, removing E234's split of
14 aligned and 16 unknown-parent rows. Each row had 32 legal component
candidates. The retained role plan finished at root accuracy 1.0, bound top-k
recall 0.7083, and count MAE 0.4390. Final total loss was 23.5988.

| Suite | n | syntax | meaningful | structure | component recall | fidelity | reward | binder applications / changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.4122 | 0.1667 | 0.1111 | 0.2497 | 3 / 1 |
| held_out | 5 | 1.0000 | 0.0000 | 0.2739 | 0.0400 | 0.0333 | 0.1398 | 3 / 1 |
| adversarial | 4 | 1.0000 | 0.2500 | 0.3556 | 0.2083 | 0.2083 | 0.3870 | 6 / 2 |
| ood | 4 | 1.0000 | 0.0000 | 0.2369 | 0.0000 | 0.0000 | 0.0000 | 0 / 0 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1371 | 0.3333 | 0.0833 | 0.4577 | 4 / 0 |

Strict five-suite compiler-tree evaluation kept syntax at 1.0, but failed nine
quality thresholds across smoke, held-out, OOD, and RICO. AgentV passed 1/5
with zero execution errors. Evaluation trace:
`0997d84c79bd74bbc69b7baf9460a7f8`.

The binder-plan-off causal ablation produced identical aggregate metrics and
gates on all five suites. Weight 1 changed four component choices across 16
applications, but none changed a suite aggregate; AgentV remained 1/5.
Ablation trace: `2070f18ae0dd27dc959e7c3ae84569b4`.

Retain the generalized binder-indexed objective and telemetry, but reject the
checkpoint. Full supervision coverage fixes E234's observability gap, yet a
component label per binder is not enough to recover meaningful structure. The
next hierarchy lever should predict grammar-derived binder topology and arity:
which binder instances are emitted and which legal parent references connect
them, rather than adding stronger component calibration or literal cases.

Machine-readable evidence:
[iter-e235-binder-instance-plan-20260716.json](iter-e235-binder-instance-plan-20260716.json).
