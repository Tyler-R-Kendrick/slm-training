# SLM-148 / SPV1-05: plan-conditioned X22 × conflict-slice campaign (slm148_fixture)

Matrix set: `slm148_x22_conflict_campaign`

Version: `spv1-05-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no production X22 checkpoint was loaded, and no ship-gate claim is made.

## Hypothesis

Plan-conditioned initialization (gold or learned) and conflict-localized recovery together lift X22 closer to acceptable programs than minimal seed or coarse remask controls, under matched seed/edit/search/verifier budgets.

## Falsifier

No seed strategy reduces seed-to-gold distance, no recovery policy improves recovery rate while preserving more correct structure than full remask, or plan features silently alter legal candidate membership.

## Seed arms

| Arm | Strategy | Seeds | Promotable | Description |
| --- | --- | --- | --- | --- |
| S0_minimal | minimal | 0,1,2 | True | Canonical minimal X22 seed; baseline for search distance. |
| S1_frequency_prior | frequency_prior | 0,1,2 | True | Deterministic train-set archetype/role frequency prior seed. |
| S2_learned_archetype_role_set | learned_archetype_role_set | 0,1,2 | True | Learned archetype + role set merged into gold topology/bindings. |
| S3_learned_full_plan | learned_full_plan | 0,1,2 | True | Fully learned plan seed (chain topology; wiring approximation). |
| S4_gold_factor_bindings | gold_factor | 0,1,2 | False | Gold binding factor substituted into predicted plan; diagnostic. |
| S5_gold_plan_oracle | gold_plan | 0,1,2 | False | Full gold plan oracle seed; diagnostic ceiling. |
| S6_retrieved_prototype | retrieved_prototype | 0,1,2 | True | Best leakage-safe retrieved valid prototype (SPV1-04 hybrid). |
| S7_plan_reranked_retrieval | plan_reranked_retrieval | 0,1,2 | True | Retrieved prototype reranked by learned plan factors. |

## Recovery arms

| Arm | Policy | Diagnostic | Description |
| --- | --- | --- | --- |
| R0_none | none | False | Canonical X22 no additional remask/recovery. |
| R1_full_remask | full_remask | False | Full/coarse remask control from SLM-113. |
| R2_suffix_rollback | suffix_rollback | False | Suffix rollback control. |
| R3_conflict_slice | conflict_slice | False | Conflict-slice localized revision. |
| R4_oracle_conflict_slice | conflict_slice_expanded | True | Oracle conflict slice expanded; diagnostic. |

## Survivors

`S0_minimal`, `S1_frequency_prior`, `S2_learned_archetype_role_set`, `S3_learned_full_plan`, `S6_retrieved_prototype`, `S7_plan_reranked_retrieval`

## Results

### S0_minimal / none / seed=0 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S0_minimal / none / seed=1 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S0_minimal / none / seed=2 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / none / seed=0 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / none / seed=1 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / none / seed=2 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / none / seed=0 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / none / seed=1 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / none / seed=2 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / none / seed=0 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / none / seed=1 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / none / seed=2 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S4_gold_factor_bindings / none / seed=0 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=gold_factor
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded
- non-promotable seed arm

### S4_gold_factor_bindings / none / seed=1 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=gold_factor
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded
- non-promotable seed arm

### S4_gold_factor_bindings / none / seed=2 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=gold_factor
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded
- non-promotable seed arm

### S5_gold_plan_oracle / none / seed=0 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 1.000
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=gold_plan
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded
- non-promotable seed arm

### S5_gold_plan_oracle / none / seed=1 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 1.000
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=gold_plan
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded
- non-promotable seed arm

### S5_gold_plan_oracle / none / seed=2 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 1.000
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=gold_plan
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded
- non-promotable seed arm

### S6_retrieved_prototype / none / seed=0 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / none / seed=1 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / none / seed=2 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / none / seed=0 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / none / seed=1 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / none / seed=2 (screening)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=none
- stage=screening
- fixture-only: no X22 model trained or decoded

### S0_minimal / none / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / full_remask / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / suffix_rollback / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / conflict_slice / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / conflict_slice_expanded / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S0_minimal / none / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / full_remask / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / suffix_rollback / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / conflict_slice / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / conflict_slice_expanded / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S0_minimal / none / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / full_remask / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / suffix_rollback / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / conflict_slice / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S0_minimal / conflict_slice_expanded / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=minimal
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S1_frequency_prior / none / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / full_remask / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / suffix_rollback / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / conflict_slice / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / conflict_slice_expanded / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S1_frequency_prior / none / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / full_remask / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / suffix_rollback / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / conflict_slice / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / conflict_slice_expanded / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S1_frequency_prior / none / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / full_remask / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / suffix_rollback / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / conflict_slice / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S1_frequency_prior / conflict_slice_expanded / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.322
- mean component coverage: 0.522
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=frequency_prior
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S2_learned_archetype_role_set / none / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / full_remask / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / suffix_rollback / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / conflict_slice / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / conflict_slice_expanded / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S2_learned_archetype_role_set / none / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / full_remask / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / suffix_rollback / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / conflict_slice / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / conflict_slice_expanded / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S2_learned_archetype_role_set / none / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / full_remask / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / suffix_rollback / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / conflict_slice / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S2_learned_archetype_role_set / conflict_slice_expanded / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.621
- mean component coverage: 0.453
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_archetype_role_set
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S3_learned_full_plan / none / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / full_remask / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / suffix_rollback / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / conflict_slice / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / conflict_slice_expanded / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S3_learned_full_plan / none / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / full_remask / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / suffix_rollback / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / conflict_slice / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / conflict_slice_expanded / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S3_learned_full_plan / none / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 2.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / full_remask / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / suffix_rollback / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / conflict_slice / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S3_learned_full_plan / conflict_slice_expanded / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.411
- mean component coverage: 0.528
- recovery rate: 0.000
- mean remasked nodes: 2.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=learned_full_plan
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S6_retrieved_prototype / none / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / full_remask / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / suffix_rollback / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / conflict_slice / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / conflict_slice_expanded / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S6_retrieved_prototype / none / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / full_remask / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / suffix_rollback / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / conflict_slice / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / conflict_slice_expanded / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S6_retrieved_prototype / none / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / full_remask / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / suffix_rollback / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / conflict_slice / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S6_retrieved_prototype / conflict_slice_expanded / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=retrieved_prototype
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S7_plan_reranked_retrieval / none / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / full_remask / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / suffix_rollback / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / conflict_slice / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / conflict_slice_expanded / seed=0 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S7_plan_reranked_retrieval / none / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / full_remask / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / suffix_rollback / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / conflict_slice / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / conflict_slice_expanded / seed=1 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

### S7_plan_reranked_retrieval / none / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 0.0
- mean preserved nodes: 3.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=none
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / full_remask / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=full_remask
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / suffix_rollback / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 3.0
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=suffix_rollback
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / conflict_slice / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=conflict_slice
- stage=cross
- fixture-only: no X22 model trained or decoded

### S7_plan_reranked_retrieval / conflict_slice_expanded / seed=2 (cross)
- records: 13
- seed valid: 13
- mean seed-to-gold ratio: 0.851
- mean component coverage: 0.387
- recovery rate: 0.000
- mean remasked nodes: 2.2
- mean preserved nodes: 1.0
- mean forwards: 64.0
- mean verifier calls: 16.0
- repeated conflict rate: 0.000
- seed_strategy=plan_reranked_retrieval
- recovery_policy=conflict_slice_expanded
- stage=cross
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic recovery arm

## Verdict

This is a fixture wiring run. It validates that the staged factorial manifest is honest (gold/oracle arms non-promotable), that plan-conditioned and retrieved seeds compile to hard-valid initial states, that the conflict-slice policy from SLM-113 can be applied to those states, and that recovery bookkeeping is deterministic and replayable. Real quality/cost claims require the trained X22 model, SLM-111 beam/depth points, and AgentV evaluation on held-out suites.
