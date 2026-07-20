# E555 — slot-pair interaction

E555 adds a multiplicative current-slot × next-slot interaction to the learned
slot-owner head. R1 crashed before completion on an empty terminal-slot HF
sequence and is not evidence; the generalized sentinel fix is regression
tested. Valid R2 processed 1,304 target tokens in 50.29 seconds under
`max_wall_minutes=3` and wrote SHA
`af53e1619e9749eab78379ae7696a929e7409dbd984a5e33481cfa050addf19e`.

OOD `n=4` fidelity is 0.3000, structure 0.1594, recall 0.1250, reward 0.5453,
and AST node F1 0.2389. This keeps E553 fidelity/reward while improving all
three topology metrics, and matches E554 topology while recovering its
fidelity/reward loss. Meaningful-v1, strict-v2, AST edge F1, and AgentV remain
zero.

**Verdict:** retain pair interaction as a Pareto lever, but reject checkpoint
promotion. Evidence: [JSON](iter-e555-slot-pair-interaction-20260720.json).
