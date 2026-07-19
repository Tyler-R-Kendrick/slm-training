# E536 — choice decision evidence

E536 closes an evaluator observability gap exposed by E535. The canonical
evaluation result already retained surface predictions and reason codes, but it
did not retain the actual choice-codec decision stream. Surface re-encoding is
not equivalent because the production codec drops unreachable declarations.

The model plugin now emits bounded `choice_decision_trace/v1` evidence aligned
to each generated record: the actual choice tokens, token count, and each
decision where a legal reference existed or a reference was selected. Evidence
contains generated decisions and decoder state only; no gold program or hidden
topology is added.

E536 holds the E531 checkpoint and every E535 OOD n=4 setting fixed. The CPU run
completed under the 170-second process cap from clean commit `acff961`, emitted
AgentEvals JSONL plus the pinned AgentV SDK bundle, and reproduced every E535
quality aggregate exactly.

| Metric | E535 | E536 | Delta |
| --- | ---: | ---: | ---: |
| Syntax parse rate | 1.0000 | 1.0000 | 0.0000 |
| Meaningful program rate v1 | 0.2500 | 0.2500 | 0.0000 |
| Placeholder fidelity | 1.0000 | 1.0000 | 0.0000 |
| Structural similarity | 0.1959 | 0.1959 | 0.0000 |
| Component type recall | 0.5417 | 0.5417 | 0.0000 |
| Reward | 0.7402 | 0.7402 | 0.0000 |
| AST node / edge F1 | 0.1627 / 0.0417 | 0.1627 / 0.0417 | 0 / 0 |
| Strict binding-aware meaning | 0.0000 | 0.0000 | 0.0000 |
| AgentV | 0 / 1 | 0 / 1 | unchanged |

The new evidence corrects E535's causal interpretation:

| Record | Choice tokens | Legal-reference decision rows | References chosen |
| --- | ---: | ---: | ---: |
| dashboard | 38 | 9 | 0 |
| gallery | 58 | 26 | 1 |
| modal | 58 | 21 | 0 |
| auth | 58 | 28 | 9 |
| **Total** | **212** | **84** | **10** |

All four streams used structural mode with no `r=` marker. E535 therefore had
zero applications because its v0.5 root-marker guard was unreachable, not
because legal reference alternatives were absent. Dashboard repeatedly chose
new components over legal references; gallery and modal also left reference
opportunities unused; auth selected references but still built an incorrect
graph.

Accept E536 as a harness improvement with no model-quality claim. The next
causal lever should target structural-mode terminal-root planning and must use
this evidence to prove reach and choice changes before training. Machine-
readable evidence is in
[the E536 JSON](iter-e536-choice-decision-evidence-20260719.json).
