# SLM-146 / SPV1-03: Plan-compiler bridge fixture matrix (slm146_fixture)

Matrix set: `slm146_semantic_plan_compiler`  
Version: `spv1-03-v1`  
Status: **fixture**  

**Claim class:** wiring / fixture only. No GPU was used, no production decoder was changed, and no ship-gate claim is made.

## Hypothesis

A deterministic plan compiler produces valid OpenUI seeds, attaches soft action features without changing legal membership, and gates hard restrictions behind certified evidence; unsafe predicted-hard controls are non-promotable diagnostics.

## Falsifier

Either the plan-derived seeds are invalid, soft features alter the legal candidate set, or any non-certified prediction removes a supported candidate in a promotable arm.

## Manifest

| Arm | Seed | Features | Restrictions | Promotable |
| --- | --- | --- | --- | --- |
| A_baseline | baseline | off | compiler_only | True |
| B_gold_seed | gold | off | compiler_only | True |
| C_gold_seed_soft | gold | soft | compiler_only | True |
| D_baseline_soft | baseline | soft | compiler_only | True |
| E_certified_restrictions | gold | soft | certified | True |
| F_unsafe_predicted_hard | gold | soft | unsafe_predicted_hard | False |

## Results

### A_baseline
- records: 13
- seed ok: 13
- seed valid: 0
- mean role coverage: 0.000
- mean topology coverage: 0.000
- mean binding coverage: 0.000
- total soft features: 0
- total hard removals: 0
- total false hard prunes: 0
- seed_mode=baseline
- feature_mode=off
- restriction_mode=compiler_only
- fixture-only: synthetic corpus, no production decoder

### B_gold_seed
- records: 13
- seed ok: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.411
- mean role coverage: 1.000
- mean topology coverage: 0.000
- mean binding coverage: 0.000
- total soft features: 0
- total hard removals: 0
- total false hard prunes: 0
- seed_mode=gold
- feature_mode=off
- restriction_mode=compiler_only
- fixture-only: synthetic corpus, no production decoder

### C_gold_seed_soft
- records: 13
- seed ok: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.411
- mean role coverage: 1.000
- mean topology coverage: 0.000
- mean binding coverage: 0.000
- total soft features: 32
- total hard removals: 0
- total false hard prunes: 0
- seed_mode=gold
- feature_mode=soft
- restriction_mode=compiler_only
- fixture-only: synthetic corpus, no production decoder

### D_baseline_soft
- records: 13
- seed ok: 13
- seed valid: 0
- mean role coverage: 0.000
- mean topology coverage: 0.000
- mean binding coverage: 0.000
- total soft features: 32
- total hard removals: 0
- total false hard prunes: 0
- seed_mode=baseline
- feature_mode=soft
- restriction_mode=compiler_only
- fixture-only: synthetic corpus, no production decoder

### E_certified_restrictions
- records: 13
- seed ok: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.411
- mean role coverage: 1.000
- mean topology coverage: 0.000
- mean binding coverage: 0.000
- total soft features: 32
- total hard removals: 13
- total false hard prunes: 0
- seed_mode=gold
- feature_mode=soft
- restriction_mode=certified
- fixture-only: synthetic corpus, no production decoder

### F_unsafe_predicted_hard
- records: 13
- seed ok: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.411
- mean role coverage: 1.000
- mean topology coverage: 0.000
- mean binding coverage: 0.000
- total soft features: 32
- total hard removals: 13
- total false hard prunes: 0
- seed_mode=gold
- feature_mode=soft
- restriction_mode=unsafe_predicted_hard
- fixture-only: synthetic corpus, no production decoder
- non-promotable diagnostic arm

## Verdict

If any promotable arm reports `false_hard_prune_count > 0`, the plan compiler has violated the certified-only restriction boundary. The unsafe predicted-hard arm is expected to show hard removals and is explicitly non-promotable.
