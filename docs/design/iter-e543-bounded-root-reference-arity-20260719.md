# E543 — bounded root-reference arity training

E543 applies the semantic section bound from E542 at training time. For each
covered record, cross entropy and accuracy now consider only arity classes that
are possible for the generated sections available when the terminal root list
closes. This removes loss pressure from impossible tokenizer-tail classes while
preserving the generalized dependency-order-aware target.

The local-only scratch continuation
`e543-e531-root-reference-bounded-r1-24s` used the same parent, corpus, recipe,
seed, batches, and 24 CPU steps as E542. It completed in 37.17 seconds under
`max_wall_minutes=3`, saw 1,270 target tokens, and wrote checkpoint SHA
`c6be3791544def59ad26b8d2b3b605a7efefd93ec83c996371e593a3251d7f90`.
All 106 non-root-head tensors are bit-identical to E542; only the root-arity
head weight and bias differ. This was an explicit `--no-sync-checkpoints`
scratch diagnostic, not a full HF-context train or promotion.

The bound materially improves head calibration. Across steps 1–12, mean
auxiliary loss falls from E542's 3.7329 to 0.8845 and mean accuracy rises from
0.0417 to 0.7500. Across steps 13–24, loss falls from 3.5496 to 0.9414 and
accuracy rises from 0.3333 to 0.5833. The active class count averages 3.71 and
4.00, respectively.

The matched four-record OOD weight-1 replay nevertheless produces exactly the
same decisions as E542's bounded replay: 11 arity applications and 7 changed
choices. Every quality metric also remains identical to E542 control: syntax
1.0, meaningful-v1 0.50, fidelity 0.5917, validity 0.7550, structure 0.3019,
component recall 0.4167, reward 0.7950, AST node F1 0.3271, and AST edge F1
0.0333. Strict binding-aware meaning remains 0.0 and AgentV fails 0/1 without
an execution error.

**Verdict:** retain bounded root-arity training as a correctness and calibration
improvement, but keep its decoder default-off and reject this checkpoint for
promotion. Better count calibration is insufficient to choose which legal
references belong in the root. The next lever should supervise reference
identity or coverage directly. Machine-readable evidence:
[JSON](iter-e543-bounded-root-reference-arity-20260719.json).
