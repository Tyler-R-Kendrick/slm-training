# E548 — semantic-role decode weight 8

E548 holds E547 checkpoint SHA
`37002bfd3c63d1ac58f5fc505bf034805b57eee2415d9e15ec1acbb81620fc57`
and its four-record OOD diagnostic recipe fixed, changing only visible
semantic-role decode weight from 4 to 8. The evaluation completed on CPU under
the 170-second hard process cap. It creates no checkpoint.

The treatment produces the same four predictions as the weight-4 control and
every headline metric is identical: syntax 1.0, meaningful-v1 0.0, fidelity
0.2583, validity 0.4550, structure 0.2248, component recall 0.2083, reward
0.5403, AST node F1 0.3270, AST edge F1 0.0, and strict-v2 0.0. AgentV remains
0/1 without execution errors.

Both weights apply the semantic-role intervention 28 times and change all 28
eligible choices. Root arity and identity telemetry is also unchanged at six
applications and two changes each. Weight 4 therefore already dominates the
eligible semantic-role choices on this subset; multiplying the same scores
cannot alter their ordering.

**Verdict:** reject scalar semantic-role weight escalation. Keep weight 4 for
the E547 diagnostic recipe and direct the next bounded experiment at learned
semantic-role candidate ordering or supervision. Machine-readable evidence:
[JSON](iter-e548-semantic-role-weight8-20260719.json).
