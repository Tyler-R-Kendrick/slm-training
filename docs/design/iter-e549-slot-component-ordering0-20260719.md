# E549 — learned slot-component ordering off

E549 holds E547 checkpoint SHA
`37002bfd3c63d1ac58f5fc505bf034805b57eee2415d9e15ec1acbb81620fc57`
and its four-record OOD diagnostic recipe fixed, changing only learned
slot-component decode weight from 4 to 0. Deterministic visible semantic-role
weight remains 4. The evaluation completed on CPU under the 170-second hard
process cap and creates no checkpoint.

Removing the learned tie-breaker improves structure from 0.2248 to 0.2713, AST
node F1 from 0.3270 to 0.3833, and AST edge F1 from 0 to 0.0625. It does not
recover fidelity: fidelity falls from 0.2583 to 0.2083 and validity from 0.4550
to 0.4250. Component recall collapses from 0.2083 to 0 and reward from 0.5403
to 0. Meaningful-v1 and strict-v2 remain 0.0; AgentV remains 0/1 without
execution errors.

The deterministic semantic-role path applies 14 times but changes no choices
without the learned head. Root-reference decoding follows the changed
trajectory, with arity applying 16 times and changing five choices and identity
applying 14 times and changing two.

**Verdict:** reject disabling learned slot-component ordering. The learned head
preserves semantic density and reward, while its full weight suppresses
topology. Test a midpoint learned weight while retaining visible-role weight 4.
Machine-readable evidence:
[JSON](iter-e549-slot-component-ordering0-20260719.json).
