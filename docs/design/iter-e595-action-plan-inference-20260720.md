# E595 — action-semantic plan inference

Date: 2026-07-20
Status: mixed positive; not promotable or ship

E595 infers `Button` in the predicted partial plan when authored prompt prose
contains action or confirmation semantics. It changes neither grammar legality
nor data-quality requirements. The matched CPU OOD `n=4` run completed within
the 170-second cap.

| Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E592 control | 0.50 / 0.00 | 0.5917 / 0.7550 | 0.4169 | 0.5417 | 0.8115 | 0.5198 / 0.3429 | 0/1 |
| `e595-e592-action-plan-r1` | 0.50 / 0.00 | 0.5917 / 0.7550 | 0.4694 | 0.6250 | 0.8115 | 0.5532 / 0.3875 | 0/1 |

The Button family is restored and structural metrics improve, but it consumes
the Modal body slot rather than confirmation, while dashboard action prose
diverts its root to Button. Family inventory without visible role binding is
therefore insufficient.

Do not promote or sync. Strict meaning-v2 remains zero and AgentV is 0/1.
Next bind inferred action families to action-role slots.

Evidence: [JSON](iter-e595-action-plan-inference-20260720.json).
