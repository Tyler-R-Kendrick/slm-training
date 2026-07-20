# E593 — optional enum-argument close score

Date: 2026-07-20
Status: mixed negative; not promotable or ship

E593 adds a default-off positive score for the already-legal close token at
optional enum-valued component arguments. It changes neither the candidate
set nor required arguments. The matched CPU OOD `n=4` ladder uses E592's
frozen local HF checkpoint and visible-context policy; every treatment
completed under the 170-second process cap.

| Run | close weight | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E592 control | 0 | 0.50 / 0.00 | 0.5917 / 0.7550 | 0.4169 | 0.5417 | 0.8115 | 0.5198 / 0.3429 | 0/1 |
| `e593-e592-enum-close2-r1` | 2 | 0.50 / 0.00 | 0.5417 / 0.6250 | 0.4231 | 0.5417 | 0.6447 | 0.5198 / 0.3429 | 0/1 |
| `e593-e592-enum-close4-r1` | 4 | 0.50 / 0.00 | 0.5833 / 0.6500 | 0.3881 | 0.4792 | 0.6573 | 0.4365 / 0.3429 | 0/1 |

Weight 2 removes the erroneous Modal size placeholder but diverts dashboard
generation from `TextContent` to an empty `Card`. Weight 4 also replaces the
gallery's invalid TextContent size with an Image and removes all reported
enum-role mismatches, but dashboard remains empty and aggregate structure,
recall, reward, and node F1 regress from E592.

Keep this generalized lever default-off and retain E592's weight-0 behavior as
the scratch baseline. Strict meaning-v2 is still zero, AgentV is 0/1 for every
arm, and no checkpoint was created or synced.

Evidence: [JSON](iter-e593-enum-close-score-20260720.json).
