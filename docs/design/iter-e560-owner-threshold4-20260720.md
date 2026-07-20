# E560 — narrow rare slot-owner coverage

E560 keeps E559's 2× exposure but narrows eligibility from owner classes with
at most 10 labels to at most 4. Five classes select 9 of 244 records, expanding
the sampling pool to 253.

The clean run processed 1,312 target tokens in 42.26 seconds under
`max_wall_minutes=3` and wrote checkpoint SHA
`dae11cee1e8fc1a2178b6397e558db9b0e4a723bbfbaf38f63af94557d7686a3`.

Against E555 on matched OOD `n=4`, structure improves 0.1594→0.2181,
component recall 0.1250→0.2083, and AST-node F1 0.2389→0.3389. Fidelity falls
0.3000→0.2583 and reward slips 0.5453→0.5403. Meaning-v1/v2 and AST-edge F1
remain 0; AgentV remains 0/1.

**Verdict:** retain the threshold-4 sampler as a topology Pareto lever without
promoting the checkpoint. Test the midpoint threshold 7 in E561. Evidence:
[JSON](iter-e560-owner-threshold4-20260720.json).
