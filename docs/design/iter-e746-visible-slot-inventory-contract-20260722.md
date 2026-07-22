# E746 — visible slot-inventory contract

**Date:** 2026-07-22  
**Decision:** retain lever contract v15 and eval harness v43; no checkpoint promotion  
**Evidence:** [`iter-e746-visible-slot-inventory-contract-20260722.json`](iter-e746-visible-slot-inventory-contract-20260722.json)

E746 corrects E745's diagnosis. The evaluator was not missing an inference rule:
the E745 recipe enabled slot-aware and semantic-role decode while
`slot_contract_in_context=false`. That made the model/evaluator contract
internally inconsistent and left strict required-inventory coverage undefined.
The fix is at the canonical lever boundary, not in metric interpretation.

Lever registry v15 now requires `slot_contract_in_context=true` for every
slot-contract decode lever and for semantic-role decode. `ModelBuildConfig` and
`TwoTowerConfig` consume that one registry and reject the former E745
combination during construction, before a run directory or evidence artifact
can exist. Eval harness v43 also includes `config.levers` in the shared suite,
scoreboard, and cache version dependency so persisted evidence identifies the
contract that admitted its recipe.

The local CPU replay uses the unchanged E735 checkpoint and the E745 recipe
with only `slot_contract_in_context false -> true`. Predictions and all model
quality metrics are unchanged. The explicitly visible request inventory makes
all three rows covered, so strict-v2 and coverage move from 0 to 1 without an
evaluator change or marker-text inference.

| Arm | Visible slots | Tokens | Parse | Meaning-v1 | Strict-v2 | Coverage | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E745 historical invalid config | no | 64 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.8308 | 0.7500 | 0.9370 | 1707 / 2551 ms | 0/1 |
| E746 valid config | yes | 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8308 | 0.7500 | 0.9370 | 1515 / 2800 ms | 0/1 |

All predictions contain only OpenUI grammar/AST tokens, schema enum literals,
and declared template markers. There are no free-form output strings. Timing is
diagnostic at `n=3`; no performance claim is made. AgentV remains 0/1, so this
is not a ship result. No model weights changed and no checkpoint was created or
promoted. The next model-quality target remains the weakest valid output shape:
hero/callout component ownership and tree exactness on a larger local suite.
