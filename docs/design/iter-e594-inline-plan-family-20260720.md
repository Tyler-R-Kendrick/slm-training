# E594 — inline semantic-plan family score

Date: 2026-07-20
Status: neutral to negative; not promotable or ship

E594 adds a default-off score for prompt-required component families that
remain missing when the decoder chooses an inline component. It counts
already-opened component tokens in the valid prefix and changes no legal
candidates. The matched CPU OOD `n=4` treatments completed within the
170-second cap.

| Run | inline weight | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E592 control | 0 | 0.50 / 0.00 | 0.5917 / 0.7550 | 0.4169 | 0.5417 | 0.8115 | 0.5198 / 0.3429 | 0/1 |
| `e594-e592-inline-plan2-r1` | 2 | 0.50 / 0.00 | 0.5917 / 0.7550 | 0.4169 | 0.5417 | 0.8115 | 0.5198 / 0.3429 | 0/1 |
| `e594-e592-inline-plan4-r1` | 4 | 0.50 / 0.00 | 0.5917 / 0.7550 | 0.4169 | 0.5417 | 0.8115 | 0.5198 / 0.3429 | 0/1 |

Weight 2 is prediction-identical to E592. Weight 4 still leaves Modal's
confirmation child as TextContent and additionally appends two raw auth
placeholders to the root. The aggregate tie therefore hides a real regression.

Keep the lever default-off and retain E592 as baseline. Strict meaning-v2
remains zero, AgentV is 0/1, and no checkpoint was created or synced.

Evidence: [JSON](iter-e594-inline-plan-family-20260720.json).
