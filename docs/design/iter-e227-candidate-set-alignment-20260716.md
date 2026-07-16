# E227 — legal-candidate compiler alignment

Status: **matched training completed; honest ship gates failed; checkpoint not
promoted**.

E226 isolated the remaining failure to learned topology/component choices while
the deterministic compiler guaranteed syntax. E227 corrected the alignment
objective to rank each gold decision only against the legal candidates returned
by the Lark/compiler forest. Candidate inventories and roles are grammar-derived;
singleton decisions still bypass inference, and no prompt, component-name, or
output-literal cases were added.

Latest `origin/main` was fetched immediately before launch. The clean E227 branch
was zero commits behind and one implementation commit ahead, so there were no
conflicts to resolve. The matched recipe retained E224's canonical judged E218
data, quota-capacity sampler, 32 CPU steps, batch 4, learning rate 0.0003, seed 0,
local frozen SmolLM2 context, lexer output, semantic-exhaustive stratification,
schema and slot context, no DESIGN context, and no checkpoint sync.

The lever executed: alignment loss fell from 15.3994 on step 1 to 2.4120 on step
32 across a mean 20.35 legal candidates (maximum 69). Final nested-component
candidate loss remained 8.2085. Total loss was 12.3030, with 22,924 prompt and
6,401 target tokens in 124.72 s; trace ID
`03eb676f6ce45fb3175278d7cac822b0`.

| Suite | n | syntax | meaningful | structure | component recall | fidelity | contract precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.0000 | 0.3094 | 0.0000 | 0.0000 | 0.0000 |
| held_out | 5 | 1.0000 | 0.0000 | 0.2739 | 0.0400 | 0.0333 | 0.2000 |
| adversarial | 4 | 1.0000 | 0.0000 | 0.2905 | 0.0000 | 0.0000 | 0.0000 |
| ood | 4 | 1.0000 | 0.0000 | 0.2369 | 0.0000 | 0.0000 | 0.0000 |
| rico_held | 3 | 1.0000 | 0.0000 | 0.0901 | 0.0000 | 0.0000 | 0.0000 |

All suites retained syntax parse 1.0 and zero fallback, but outputs collapsed to
mostly empty `Stack` layouts. Twelve ship gates failed and AgentV passed 0/5
without execution errors, regressing from E226's five failures and 1/5 pass.
Restricted candidate CE is therefore falsified as a sufficient topology
objective. The next experiment should supervise a positive margin between the
gold populated-child branch and the legal empty-list alternative, derived from
the same forest; repeating this loss or adding literal cases is not justified.

Machine-readable evidence:
[iter-e227-candidate-set-alignment-20260716.json](iter-e227-candidate-set-alignment-20260716.json).
