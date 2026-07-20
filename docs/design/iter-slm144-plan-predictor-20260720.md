# SLM-144 / SPV1-01: Archetype + role-set predictor fixture matrix (slm144_fixture)

Matrix set: `slm144_plan_predictor`  
Version: `spv1-01-v1`  
Status: **fixture**  

**Claim class:** wiring / fixture only. No GPU was used, no production TwoTower wiring was touched, and no ship-gate claim is made.

## Manifest

Hypothesis: A tiny CPU predictor can learn archetype and role-set factors from a component-family count vector, and learned heads outperform frequency baselines on a controlled fixture corpus.

| Arm | Archetype | Role set | Status |
| --- | --- | --- | --- |
| baseline_none | none | none | fixture |
| frequency_prior | frequency | frequency | fixture |
| serialized_inventory | none | serialized | fixture |
| set_matching | none | predicted | fixture |
| gold_archetype | gold | predicted | fixture |
| gold_role_set | predicted | gold | fixture |
| gold_both | gold | gold | fixture |

## Results

### baseline_none
- archetype accuracy: 0.0
- role precision: 0.0
- role recall: 0.0
- role F1: 0.0
- archetype_source=none
- role_set_source=none
- fixture-only: tiny CPU net, no real checkpoint

### frequency_prior
- archetype accuracy: 0.23076923076923078
- role precision: 0.7307692307692307
- role recall: 1.0
- role F1: 0.8444444444444443
- archetype_source=frequency
- role_set_source=frequency
- fixture-only: tiny CPU net, no real checkpoint

### serialized_inventory
- archetype accuracy: 0.0
- role precision: 0.7307692307692307
- role recall: 1.0
- role F1: 0.8444444444444443
- inventory token accuracy: 0.7307692170143127
- archetype_source=none
- role_set_source=serialized
- fixture-only: tiny CPU net, no real checkpoint
- training wall time: 11544.4 ms

### set_matching
- archetype accuracy: 0.0
- role precision: 1.0
- role recall: 1.0
- role F1: 1.0
- archetype_source=none
- role_set_source=predicted
- fixture-only: tiny CPU net, no real checkpoint
- training wall time: 11544.4 ms

### gold_archetype
- archetype accuracy: 1.0
- role precision: 1.0
- role recall: 1.0
- role F1: 1.0
- archetype_source=gold
- role_set_source=predicted
- fixture-only: tiny CPU net, no real checkpoint
- training wall time: 11544.4 ms

### gold_role_set
- archetype accuracy: 0.9230769230769231
- role precision: 1.0
- role recall: 1.0
- role F1: 1.0
- archetype_source=predicted
- role_set_source=gold
- fixture-only: tiny CPU net, no real checkpoint
- training wall time: 11544.4 ms

### gold_both
- archetype accuracy: 1.0
- role precision: 1.0
- role recall: 1.0
- role F1: 1.0
- archetype_source=gold
- role_set_source=gold
- fixture-only: tiny CPU net, no real checkpoint

## Verdict

Fixture wiring only. The arms exercise baseline, frequency, learned serialized-inventory, learned set-matching, and oracle upper-bound code paths on a deterministic 64-example corpus. Generalization to real OpenUI programs requires a trained model, held-out suites, and honest ship-gate evaluation.
