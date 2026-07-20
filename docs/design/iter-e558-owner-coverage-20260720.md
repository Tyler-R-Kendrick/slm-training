# E558 — rare slot-owner coverage

E558 adds deterministic, auditable record oversampling to the canonical train
loop. Owner classes with at most 10 visible-slot labels selected 75 of 244
records; fourfold exposure expanded the sampling pool to 469 records.

The clean run processed 1,222 target tokens in 43.74 seconds under
`max_wall_minutes=3` and wrote checkpoint SHA
`a45909dffd103df353bff944aedbafa1e386b2bf657c5dc02f2d956e06381ede`.
The preceding 43.31-second r1 engineering trial completed, but its dirty-tree
version stamp excludes it from the decision.

Against E555 on the matched OOD `n=4` diagnostic, fidelity improves
0.3000→0.4250 and meaningful-v1 reaches 0.25. The treatment overcorrects:
structure falls 0.1594→0.0921, reward 0.5453→0.4075, and AST-node F1
0.2389→0.1393. Binding-aware meaningful-v2 stays 0 and AgentV stays 0/1.

**Verdict:** retain the generalized sampler, reject the 4× checkpoint, and test
a gentler 2× exposure in E559. Evidence:
[JSON](iter-e558-owner-coverage-20260720.json).
