# SLM-233: matched recursive-depth campaign

Status: **bounded_matched_proxy_complete**
Verdict: **architecture_not_identifiable**
Claim class: `architecture_not_identifiable_not_ship`

## Gate-conditioned scope

The semantic floor is not escaped, SLM-230 is stagnant, SLM-231 is expansive-unstable, and SLM-232 is unstable. SLM-243 permits only the LayerScale diagnostic through R=8. Therefore this is the required bounded proxy/control campaign, not a semantic architecture or promotion experiment.

| gate | verdict | scientific hash |
| --- | --- | --- |
| dynamics | `expansive_unstable` | `06bfd4ecd68f936b4c548b2a7a5fad39bbe56e0bdf76dc7308477635bb2e6913` |
| floor | `inconclusive` | `7839ef6b6e37710d487757da9170017d7b76a9d12ca1fb314bdb0fa23a4dd83d` |
| observability | `stagnant` | `7e9534057fa22bd041366f62cd1ba24e02c97b3b3095b4d726601f17063a8cbc` |
| update | `layerscale_preferred` | `4b5f1605db3064e98b4fcdd24c1ad24e7853745b5a86c8d175b1f0645d6b6a26` |
| z_use | `unstable` | `10361c3e225b4e7d5669e14665af93170e38717831e9d101e87b2aa5a970dce6` |

## Matched primary matrix

All A-E arms execute exactly four transformer blocks per denoiser call, use three paired seeds, identical records/order/corruption, common initialization hashes, and the same decode/evaluator budget.

| arm | description | train R | params | bytes | heldout NLL | wall s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | stacked final-only | 1 | 34290 | 178186 | 12.169158 | 1.529 |
| B | shared recursive z final-only | 2 | 26914 | 141493 | 12.971812 | 1.753 |
| C | shared recursive z normalized all-depth | 2 | 26914 | 141493 | 12.970454 | 2.445 |
| D | shared y-only normalized all-depth | 2 | 25618 | 135486 | 12.965023 | 3.147 |
| E | shared recursive z R4 normalized all-depth | 4 | 22562 | 112314 | 12.965347 | 3.873 |

## Fairness and secondary parameter view

- Fairness manifest: `f9055c2635f80ef6f0e8c4ba25df39934419bf2f40ab803192aaee75d920f155`
- Raw matrix: `e6d6e2a947f5689620c4b2cbcff266507875548463f7e683293b1976db77ef10` (21 train cells; 60 test-depth cells)
- P-D minus P-A active-parameter residual: `0`
- P-D minus P-A total-parameter residual: `32` (the frozen LayerScale vectors remain serialized and are not hidden)

## Outcomes and claim boundary

- Train/held-out NLL, bounded free-running parse/structure/reward, block/FLOP/parameter/byte/wall accounting, and LayerScale R-test diagnostics were measured.
- Protected semantic and recovery/compositional outcomes are censored, not encoded as zero, because the floor gate does not authorize them.
- Prior SLM-230/231/232 numeric rows were not transplanted to these scratch states; only their authoritative gate verdicts were joined.
- No durable checkpoint was created, no model card update is triggered, and no production default changed.

## RecursiveCoreGateV2

- Verdict: **architecture_not_identifiable**
- Allowed work: `["bounded_proxy_controls", "architecture_repair_without_semantic_claim"]`
- Blocked claims: `["semantic_architecture_efficacy", "explicit_z_mechanism", "rsc3", "rsc4", "checkpoint_promotion", "production_default_change", "ship"]`
- Checkpoint refs: `[]`
- Rationale: the semantic floor did not escape, so finite matched proxy movement cannot identify a semantic architecture effect

This verdict is not `no_recursive_gain`: the architecture effect is unidentifiable under the current semantic floor.

## AgentEvals / AgentV

- SDK: `@agentv/core`
- Summary: `{"durationMs": 26, "executionErrors": 0, "failed": 0, "meanScore": 1, "passed": 6, "total": 6}`

Report hash: `5e174941d5e01aad07fe7ef3a0812e4564486fef35562a69f06a7a1bdf96abc2`
