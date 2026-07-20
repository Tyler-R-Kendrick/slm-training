# E551 — slot lexeme prior off

E551 repeats E547's 24-step multiplier-2 continuation from E544, changing only
the corpus-derived slot lexeme prior weight from 1 to 0. The CPU HF-context run
processed 1,304 target tokens in 41.85 seconds under `max_wall_minutes=3` and
wrote local checkpoint SHA
`e7921e66df8d2c76b96c1577c7cfb3b35c97879d69f44da0c41ab787dac32fc6`.

On matched OOD `n=4`, fidelity improves 0.2583→0.3000, validity 0.455→0.480,
and reward 0.5403→0.5453. Structure regresses 0.2248→0.1594, component recall
0.2083→0.1250, and AST node F1 0.3270→0.2389. Meaningful-v1, strict-v2, AST
edge F1, and AgentV remain zero.

**Verdict:** reject prior removal and the checkpoint. The lexeme prior carries
useful topology/density signal but is overconfident; calibrate or regularize it
rather than deleting it. Evidence: [JSON](iter-e551-slot-lexeme-prior0-20260719.json).
