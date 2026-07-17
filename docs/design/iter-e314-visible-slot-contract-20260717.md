# E314 visible slot-contract training data — 2026-07-17

Honest evaluation receives a production `GenerationRequest.slot_contract` and
surfaces it into the model-visible prompt. E307 training did not consistently
use that request shape. E314 rebuilds E307 through the existing
`prompt_slot_contract` transform so training and evaluation expose the same
declared slot inventory.

The first build, v1, revealed a harness bug: `ensure_prompt_inventory` returned
early when a prompt contained any placeholder. As a result, 140 edit/repair
prompts exposed only partial old-source inventories. V1 was rejected and
removed. The coverage-aware fix now returns early only when the prompt already
covers every declared slot; its idempotence and partial-contract behavior are
covered by tests.

## Accepted v2

| Measure | Result |
| --- | ---: |
| Rows | 592 |
| Full visible declared contract | 592/592 |
| Targets + declared slots equal E307 | 592/592 |
| Prompt chars p95 / max | 379 / 485 |
| Independent-verifier rejects | 7 |
| Build errors | 0 |
| Eval contamination | 0/19 |
| Target tokens p95 / max | 93 / 112 |
| Targets over 256 tokens | 0 |
| AgentV diagnostic | 1/1 |

The build reproduced at content fingerprint
`d099d6ae53a01685de5134b9cffd97e7aaa6e9c5c272ebb67d08f074c1c128f3`.
All five suites retain component occurrence and type coverage 1.0.

Published snapshot:
`src/slm_training/resources/data/train/e314_visible_slot_contract_v2`.

**Verdict:** accept v2 for a matched 20k-token train. This is deterministic
data-build evidence, not checkpoint quality or a ship claim.
