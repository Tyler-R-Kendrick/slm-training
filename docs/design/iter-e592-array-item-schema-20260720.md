# E592 — typed component-array item legality

Date: 2026-07-20
Status: positive diagnostic; not promotable or ship

E592 preserves component-array item schemas in the canonical choice decoder.
Raw placeholders are no longer legal children where the schema requires
component expressions. Arrays without an item contract keep their prior
behavior, and component-family enumeration is not over-constrained because
canonical gold legitimately uses direct component children.

The treatment uses E591's weight-2 recipe: CPU, frozen local HF context, honest
visible slot/role contracts, constrained LTR, 8 steps, 4 attempts, and a
160-token canvas. It completed under 170 seconds on OOD `n=4`.

Two rerun attempts were discarded before the final matched result: one failed
configuration validation because visible role context was omitted; one
completed with slot-contract constrained decode accidentally omitted and was
therefore non-comparable. Neither is used as evidence. The final result below
matches E591's persisted evaluation policy exactly.

| Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E591 control | 0.25 / 0.00 | 0.5917 / 0.7550 | 0.4044 | 0.4583 | 0.8085 | 0.5198 / 0.3250 | 0/1 |
| `e592-e591-array-items-r1` | 0.50 / 0.00 | 0.5917 / 0.7550 | 0.4169 | 0.5417 | 0.8115 | 0.5198 / 0.3429 | 0/1 |

Modal's raw repeated placeholder children are replaced by component
expressions. Meaningful-v1 doubles, while structure, component recall, reward,
and AST-edge F1 improve. Remaining errors include an extra Modal size value,
incorrect child families, and the two low-recall dashboard/gallery records.

Keep typed-array legality as the next scratch baseline. Do not promote or sync:
strict meaning-v2 remains zero, AgentV is 0/1, and this is only four OOD rows.

Evidence: [JSON](iter-e592-array-item-schema-20260720.json).
