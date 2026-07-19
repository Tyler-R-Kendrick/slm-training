# E552 — half-strength slot lexeme prior

E552 repeats the matched E547/E551 24-step continuation, changing only the
slot lexeme prior weight to 0.5. It processed 1,304 target tokens in 34.75
seconds under `max_wall_minutes=3` and wrote local checkpoint SHA
`49a9c1119d28f95437f86cfca5f8c06467173d56d1e060757cea8af0a151fc04`.

OOD `n=4` fidelity is 0.1333, validity 0.2800, structure 0.2181, component
recall 0.1250, reward 0.3435, and AST node F1 0.3389. Meaningful-v1,
strict-v2, AST edge F1, and AgentV remain zero. Relative to weight 1, fidelity,
recall, reward, and structure all regress.

**Verdict:** reject weight 0.5 and the checkpoint. The `0/0.5/1` ladder is
non-monotonic, so close scalar prior tuning and change supervision or prior
construction. Evidence: [JSON](iter-e552-slot-lexeme-prior05-20260719.json).
