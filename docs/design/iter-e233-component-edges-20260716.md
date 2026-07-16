# E233 — resolved-AST component edges

Date: 2026-07-16
Status: completed; checkpoint rejected; not promotable or ship

E233 adds prompt-conditioned parent→child component-edge supervision on top of
E232's grammar-role root/count plan. Targets are traversed from the official
parser's resolved AST. At decode time, the compiler's binder declaration and
reference roles identify known parent components for the active declaration;
edge logits only reorder `component_bound` candidates already admitted by the
completion forest. The implementation contains no component-name, prompt,
output, or fixture-layout cases and cannot change grammar legality.

Immediately before training, the isolated branch fetched and rebased twice onto
`origin/main` at `59807a1`, was zero commits behind, and was clean. The first
launch stopped before model construction because the shared editable virtualenv
resolved the shared checkout; the official run used `PYTHONPATH=src`, and 78
focused tests plus Ruff passed against the isolated worktree code.

The matched E232 recipe used the published 126-row
`e230_diverse_judged_roots_v2` corpus, CPU, 32 steps, batch 4, learning rate
0.0003, seed 0, frozen local SmolLM2-135M, lexer output, exhaustive compiler
CE/margin 1.0, role/count loss and decode weight 1.0, edge loss and decode weight
1.0, schema and train-only slot context, no DESIGN context, capacity-aware
sampling, and no checkpoint sync. The 128 draws covered 81 unique rows, including
30 RICO and 25 human-curated draws. Training took 179.08 s; trace:
`582fb34d2260314b0540c003e3028212`; checkpoint SHA:
`46141ac11750d7525140c8d987c5fad83ab430fda7a3fb28a1cf4a31f9cf2575`.

The edge target learned on sampled train batches: balanced BCE fell 1.4627 →
0.3626 and gold-count top-k recall rose 0 → 0.5000. The retained role plan also
reproduced E232's final root accuracy 1.0, bound top-k recall 0.7083, and count
MAE 0.4357. Final total loss was 21.4669.

| Suite | n | syntax | meaningful | structure | component recall | fidelity | reward | edge applications / changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.4642 | 0.2500 | 0.5278 | 0.8073 | 9 / 0 |
| held_out | 5 | 1.0000 | 0.0000 | 0.3369 | 0.1567 | 0.2800 | 0.7330 | 15 / 1 |
| adversarial | 4 | 1.0000 | 0.2500 | 0.3895 | 0.2083 | 0.4167 | 0.6148 | 8 / 0 |
| ood | 4 | 1.0000 | 0.0000 | 0.3750 | 0.2083 | 0.2583 | 0.7265 | 9 / 0 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1628 | 0.4444 | 0.1250 | 0.6865 | 6 / 0 |

Strict five-suite compiler-tree evaluation kept syntax at 1.0 but failed four
frontier thresholds: smoke, held-out, and OOD meaningful-program rates plus RICO
structure. AgentV passed 1/5 with zero execution errors. Evaluation trace:
`4e82a613e3674701fe5006a0eafb34c4`.

The edge-off causal ablation produced identical aggregate metrics and gates on
all five suites. Although edge weight 1 changed one held-out component choice
across 47 applications, it did not change a suite aggregate; AgentV remained
1/5. Ablation trace: `decd0dc84f2e2255fe95767d1a5a486e`.

Retain the generalized AST-edge target, compiler-role parent lookup, telemetry,
and causal override, but reject the checkpoint. The auxiliary target learns yet
does not causally improve this frontier. A next lever should model binder-level
instance topology or directly align edge decisions at their decode states,
rather than increasing pooled edge calibration.

Machine-readable evidence:
[iter-e233-component-edges-20260716.json](iter-e233-component-edges-20260716.json).
