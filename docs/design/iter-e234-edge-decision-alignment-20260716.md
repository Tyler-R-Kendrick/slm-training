# E234 — parent-conditioned edge decision alignment

Date: 2026-07-16
Status: completed; checkpoint rejected; not promotable or ship

E234 replaces E233's global edge-matrix BCE with direct cross-entropy at each
grammar-derived `component_bound` decision. The active declaration's parent
component comes from the compiler token-role reference graph; the target is the
gold component among only the candidates admitted by that decision's compiler
forest. It shares the exact scorer used at decode, contains no component-name or
fixture-layout cases, and cannot change grammar legality.

Immediately before training, the isolated branch fetched and rebased onto
`origin/main` at `f148a9d`, was zero commits behind, and was clean. Ruff, compile
checks, and 78 focused tests passed against the isolated source.

The matched E233 recipe used the published 126-row
`e230_diverse_judged_roots_v2` corpus, CPU, 32 steps, batch 4, learning rate
0.0003, seed 0, frozen local SmolLM2-135M, lexer output, exhaustive compiler
CE/margin 1.0, role/count loss and decode weight 1.0, direct edge-alignment loss
and decode weight 1.0, global edge BCE 0, schema and train-only slot context, no
DESIGN context, capacity-aware sampling, and no checkpoint sync. The 128 draws
covered 81 unique rows, including 30 RICO and 25 human-curated draws. Training
took 175.12 s; trace: `c65bd85e448ca8b1da329f9a66abb725`; checkpoint SHA:
`350b7c5c1d08a26abf0bd26ff440edb95317243c91b3f6f14e77c36d62a0fc68`.

Direct legal-decision accuracy rose 0 → 0.5714 and loss fell 3.7061 → 1.5264.
The final sampled batch had 14 aligned bound decisions, 16 bound decisions with
no parent recoverable from the partial prefix, and 32 legal component candidates
per aligned row. The retained role plan finished at root accuracy 1.0, bound
top-k recall 0.7083, and count MAE 0.4331. Final total loss was 22.6982.

| Suite | n | syntax | meaningful | structure | component recall | fidelity | reward | edge applications / changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.4642 | 0.2500 | 0.5278 | 0.8073 | 9 / 0 |
| held_out | 5 | 1.0000 | 0.0000 | 0.3369 | 0.1567 | 0.2800 | 0.7330 | 16 / 1 |
| adversarial | 4 | 1.0000 | 0.2500 | 0.3619 | 0.2083 | 0.2917 | 0.5743 | 9 / 3 |
| ood | 4 | 1.0000 | 0.0000 | 0.3750 | 0.2083 | 0.2583 | 0.7265 | 12 / 0 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1628 | 0.4444 | 0.1250 | 0.6865 | 7 / 1 |

Strict five-suite compiler-tree evaluation kept syntax at 1.0 but failed the
same four frontier thresholds: smoke, held-out, and OOD meaningful-program rates
plus RICO structure. AgentV passed 1/5 with zero execution errors. Evaluation
trace: `43dc458ea59cd8975ec58339226d2dbb`.

The edge-off causal ablation produced identical aggregate metrics and gates on
all five suites. Weight 1 changed five component choices across 53 applications,
but none changed a suite aggregate; AgentV remained 1/5. Ablation trace:
`f7e069c4c853243c3bfa1e84552826fa`.

Retain the generalized decision-alignment objective and telemetry, but reject
the checkpoint. Stronger alignment alone is not the answer. The high
unknown-parent count indicates the next hierarchy lever must represent
binder-level instance topology across declaration order, rather than adding
type-edge calibration.

Machine-readable evidence:
[iter-e234-edge-decision-alignment-20260716.json](iter-e234-edge-decision-alignment-20260716.json).
