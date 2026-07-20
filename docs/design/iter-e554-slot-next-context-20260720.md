# E554 — next-slot context for slot ownership

E554 changes only the slot-owner architecture: each slot is encoded with the
next visible slot during training and decode. The valid R2 run processed 1,304
target tokens in 39.91 seconds under `max_wall_minutes=3` and wrote local
checkpoint SHA
`af3cbce7ca8c2adfbccc8d5ad0550361e2c30f56a6da04f6390615d40c67b579`.
R1 is excluded because its summary did not persist the treatment flag.

OOD `n=4` fidelity is 0.2583, validity 0.4550, structure 0.1594, component
recall 0.1250, reward 0.5328, and AST node F1 0.2389. Versus E553, structure,
recall, and AST node F1 improve, while fidelity and reward regress.
Meaningful-v1, strict-v2, AST edge F1, and AgentV remain zero.

**Verdict:** reject the checkpoint. Next-slot context partly restores topology
but does not resolve the quality tradeoff. Evidence:
[JSON](iter-e554-slot-next-context-20260720.json).
