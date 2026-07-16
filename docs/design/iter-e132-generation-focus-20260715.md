# E132 generation-focused mixture — 2026-07-15

E132 emphasizes generation tasks at 60% while retaining 20% patch/edit and 20%
repair/completion supervision. It uses the 405-record judged corpus and the
E127 schema/slot recipe.

| Suite | n | Parse | Placeholder validity | Structural similarity | Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.0 | 0.0 | 0.1742 | 0.0 |

All three prompts (hero, button, callout) failed parse and placeholder checks.
Training loss finished at **12.55**, with train telemetry and a complete
three-prompt AgentEvals bundle persisted. The generation-focused mixture is
rejected; task imbalance alone did not explain the failure.
