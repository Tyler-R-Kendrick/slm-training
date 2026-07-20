# E556 — combined slot context

E556 combines E554 next-slot text with E555 multiplicative slot-pair
interaction. It processed 1,304 target tokens in 68.42 seconds under
`max_wall_minutes=3` and wrote SHA
`139c670c7e1d087101111720fbb458f2a0ad1b3284e9d57fa3eff4fa95831f0a`.

OOD `n=4` structure 0.1594, recall 0.1250, and AST node F1 0.2389 merely match
the single-factor treatments, while fidelity falls to 0.2167 and reward to
0.5203. Meaningful-v1, strict-v2, AST edge F1, and AgentV remain zero.

**Verdict:** reject the combined checkpoint, retain E555 alone, and close the
context factorial. Evidence:
[JSON](iter-e556-slot-context-combined-20260720.json).
