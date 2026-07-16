# E236 — binder-reference topology

Date: 2026-07-16
Status: completed; checkpoint rejected; not promotable or ship

E236 adds a prompt-conditioned parent-binder × child-binder head. Targets come
only from Lark/compiler-classified `bind_reference*` decisions; the parent is
the grammar-native active declaration binder, and cross-entropy is restricted
to compiler-legal binder candidates. Decode bias uses the same head and legal
candidate set. It cannot alter grammar legality and contains no fixture,
component-name, or literal-layout cases.

Immediately before training, the isolated branch fetched and rebased onto
`origin/main` at `f5ccdd6`, was zero commits behind, and was clean. Ruff,
compile checks, and 80 focused tests passed with the official OpenUI bridge.

The matched E235 recipe used the published 126-row
`e230_diverse_judged_roots_v2` corpus, CPU, 32 steps, batch 4, learning rate
0.0003, seed 0, frozen local SmolLM2-135M, lexer output, exhaustive compiler
CE/margin 1.0, role/count loss and decode weight 1.0, topology loss and decode
weight 1.0, all component-edge and binder-component-plan weights 0, schema and
train-only slot context, no DESIGN context, capacity-aware sampling, and no
checkpoint sync. Training took 149.42 s; trace:
`6e771f4bdcac1e9720595da6f3d75840`; checkpoint SHA:
`94e1d042c56443764639b31cccae3340f4263fa2919db5a59fa933c56fef8c43`.

The topology objective did not learn on the matched run. First-to-last sampled
batch loss rose 1.1498 → 1.3503 and accuracy moved 0.5455 → 0.5238 while rows
changed 11 → 21 and mean legal candidates changed 5.0 → 7.7619. The retained
role plan ended at root accuracy 1.0, bound top-k recall 0.7083, and count MAE
0.4603. Final total loss was 28.3277.

| Suite | n | syntax | meaningful | structure | component recall | fidelity | reward | topology applications / changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.0000 | 0.3094 | 0.0000 | 0.0000 | 0.0000 | 6 / 0 |
| held_out | 5 | 1.0000 | 0.0000 | 0.2514 | 0.0000 | 0.0000 | 0.0000 | 10 / 0 |
| adversarial | 4 | 1.0000 | 0.0000 | 0.2905 | 0.0000 | 0.0000 | 0.0000 | 8 / 0 |
| ood | 4 | 1.0000 | 0.0000 | 0.2369 | 0.0000 | 0.0000 | 0.0000 | 8 / 0 |
| rico_held | 3 | 1.0000 | 0.0000 | 0.0901 | 0.0000 | 0.0000 | 0.0000 | 6 / 0 |

Strict five-suite compiler-tree evaluation kept syntax at 1.0, but meaningful
program, component recall, fidelity, and reward collapsed to 0 throughout.
Twelve thresholds failed and AgentV passed 0/5 with zero execution errors.
Evaluation trace: `61e2e55354eaeb64023169fc8f193dd6`.

The topology-off decode ablation was identical on every aggregate metric and
gate. Weight 1 was applied at 38 legal binder-reference decisions but changed
no choices, so the quality collapse is attributable to training interference,
not topology reranking. AgentV remained 0/5. Ablation trace:
`8add5fcc593118603b959abebbb49bd1`.

Retain the generalized grammar-derived head and telemetry, but reject the
checkpoint and direct weight-1 objective. A future topology experiment must
first establish a stable learning signal with normalized or staged loss and
must separately model reference arity/stop decisions; increasing this weight or
adding literal cases is unsupported.

Machine-readable evidence:
[iter-e236-binder-topology-20260716.json](iter-e236-binder-topology-20260716.json).
