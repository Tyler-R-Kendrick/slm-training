# E226 — honest compiler policy and parse metric correction

Status: **existing E224 checkpoint reevaluated; syntax parse 1.0 on all five
suites; honest ship gates failed on meaningful-program quality; no training or
checkpoint promotion**.

E225 was not a valid test of the intended production request path. The evaluator
preferred `generate_with_stats(prompt)` over `generate_batch_requests(request)`,
dropping the request schema and exact slot contract while collecting telemetry.
It also reported `parse_rate` as an alias of the meaningful-program heuristic,
which made structurally valid compiler output look like a parser failure.

The generalized repair keeps telemetry around the production request API,
separates syntax parse from meaningful-program quality, and makes ship evaluation
honest by default. The compiler tree now treats a generated-AST document boundary
separately from unresolved binder references, excludes postfix expression paths
after a complete root, reserves one grammar-draft window beyond the learned length
estimate, and selects EOS near the cap once the AST and references are complete.
All restrictions derive from Lark state, generated AST/schema state, and the
request contract; no component/output string cases were added.

Recipe: CPU, existing 32-step HF-context E224 checkpoint, local-only SmolLM2
context, `eval:remediated`, lexer output, compiler tree, schema + visible slot
contract in context, no DESIGN context, no unconstrained fallback. `--ship-gates`
made `honest_slot_contract=true` and `slot_contract_constrained_decode=true`.

| Suite | n | syntax parse | meaningful program | structure | component recall | fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.0000 | 0.3628 | 0.2500 | 0.7222 |
| held_out | 5 | 1.0000 | 0.0000 | 0.2309 | 0.1567 | 0.4533 |
| adversarial | 4 | 1.0000 | 0.0000 | 0.2982 | 0.4583 | 0.7500 |
| ood | 4 | 1.0000 | 0.0000 | 0.2762 | 0.2083 | 0.4250 |
| rico_held | 3 | 1.0000 | 0.3333 | 0.2380 | 0.4444 | 0.1667 |

Every suite has contract precision 1.0, zero compiler fallback, zero unconstrained
fallback, and zero full-vocabulary projections. AgentV passed 1/5 with no execution
errors. Five honest gates still fail: meaningful-program rate on smoke, held,
adversarial, and OOD, plus held structural similarity. The remaining dominant
failures are trivial layouts and low component recall, so the next matched train
must target topology/component branch supervision; more syntax repair or generic
sampling exposure is not justified by this evidence.

Machine-readable evidence:
[iter-e226-honest-compiler-policy-20260716.json](iter-e226-honest-compiler-policy-20260716.json).

