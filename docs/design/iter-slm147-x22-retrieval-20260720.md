# SLM-147 / SPV1-04: X22 leakage-safe prototype retrieval (slm147_fixture)

Matrix set: `slm147_x22_retrieval`

Version: `spv1-04-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no production X22 checkpoint was loaded, and no ship-gate claim is made.

## Hypothesis

A leakage-safe retrieved hard-valid AST prototype initializes X22 closer to an acceptable target than the generic minimal seed. SemanticPlan-aware and AST-sketch retrieval select more editable prototypes than surface prompt or random controls.

## Falsifier

Retrieved prototypes are no closer to gold than the minimal seed, simple prompt/component retrieval matches or beats plan/sketch retrieval, or adaptation/remapping fails closed too often for the strategy to be useful.

## Manifest

| Arm | Strategy | Seeds | Promotable | Description |
| --- | --- | --- | --- | --- |
| A_minimal_seed | minimal | 0,1,2 | True | Canonical minimal X22 seed; baseline for search distance. |
| B_random_prototype | random | 0,1,2 | True | Random valid training prototype control. |
| C_prompt_similarity | prompt_similarity | 0,1,2 | True | Prototype retrieved by prompt-token overlap. |
| D_ast_sketch | ast_sketch | 0,1,2 | True | Prototype retrieved by binding-aware AST sketch. |
| E_semantic_plan | semantic_plan | 0,1,2 | True | Prototype retrieved by SemanticPlanV1 factor fingerprints. |
| F_hybrid | hybrid | 0,1,2 | True | Weighted hybrid of prompt, AST-sketch, and plan similarity. |
| G_oracle_nearest | oracle_nearest | 0,1,2 | False | Oracle nearest training prototype; diagnostic ceiling only. |
| H_retrieval_as_context | retrieval_as_context | 0,1,2 | True | Historical control: retrieval added as context, seed remains minimal. |

## Results

### A_minimal_seed (seed=0)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.621
- mean component coverage: 0.453
- leakage pass: 13
- adaptation valid: 0
- strategy=minimal
- k=1
- fixture-only: no X22 model trained or decoded

### A_minimal_seed (seed=1)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.621
- mean component coverage: 0.453
- leakage pass: 13
- adaptation valid: 0
- strategy=minimal
- k=1
- fixture-only: no X22 model trained or decoded

### A_minimal_seed (seed=2)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.621
- mean component coverage: 0.453
- leakage pass: 13
- adaptation valid: 0
- strategy=minimal
- k=1
- fixture-only: no X22 model trained or decoded

### B_random_prototype (seed=0)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.717
- mean component coverage: 0.594
- leakage pass: 13
- adaptation valid: 8
- mean retrieval score: 0.000
- strategy=random
- k=1
- fixture-only: no X22 model trained or decoded

### B_random_prototype (seed=1)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.690
- mean component coverage: 0.523
- leakage pass: 13
- adaptation valid: 7
- mean retrieval score: 0.000
- strategy=random
- k=1
- fixture-only: no X22 model trained or decoded

### B_random_prototype (seed=2)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.729
- mean component coverage: 0.435
- leakage pass: 13
- adaptation valid: 9
- mean retrieval score: 0.000
- strategy=random
- k=1
- fixture-only: no X22 model trained or decoded

### C_prompt_similarity (seed=0)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.656
- mean component coverage: 0.618
- leakage pass: 13
- adaptation valid: 8
- mean retrieval score: 0.846
- strategy=prompt_similarity
- k=1
- fixture-only: no X22 model trained or decoded

### C_prompt_similarity (seed=1)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.656
- mean component coverage: 0.618
- leakage pass: 13
- adaptation valid: 8
- mean retrieval score: 0.846
- strategy=prompt_similarity
- k=1
- fixture-only: no X22 model trained or decoded

### C_prompt_similarity (seed=2)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.656
- mean component coverage: 0.618
- leakage pass: 13
- adaptation valid: 8
- mean retrieval score: 0.846
- strategy=prompt_similarity
- k=1
- fixture-only: no X22 model trained or decoded

### D_ast_sketch (seed=0)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.744
- mean component coverage: 0.565
- leakage pass: 13
- adaptation valid: 7
- mean retrieval score: 0.000
- strategy=ast_sketch
- k=1
- fixture-only: no X22 model trained or decoded

### D_ast_sketch (seed=1)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.744
- mean component coverage: 0.565
- leakage pass: 13
- adaptation valid: 7
- mean retrieval score: 0.000
- strategy=ast_sketch
- k=1
- fixture-only: no X22 model trained or decoded

### D_ast_sketch (seed=2)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.744
- mean component coverage: 0.565
- leakage pass: 13
- adaptation valid: 7
- mean retrieval score: 0.000
- strategy=ast_sketch
- k=1
- fixture-only: no X22 model trained or decoded

### E_semantic_plan (seed=0)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.851
- mean component coverage: 0.387
- leakage pass: 13
- adaptation valid: 13
- mean retrieval score: 0.631
- strategy=semantic_plan
- k=1
- fixture-only: no X22 model trained or decoded

### E_semantic_plan (seed=1)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.851
- mean component coverage: 0.387
- leakage pass: 13
- adaptation valid: 13
- mean retrieval score: 0.631
- strategy=semantic_plan
- k=1
- fixture-only: no X22 model trained or decoded

### E_semantic_plan (seed=2)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.851
- mean component coverage: 0.387
- leakage pass: 13
- adaptation valid: 13
- mean retrieval score: 0.631
- strategy=semantic_plan
- k=1
- fixture-only: no X22 model trained or decoded

### F_hybrid (seed=0)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.851
- mean component coverage: 0.387
- leakage pass: 13
- adaptation valid: 13
- mean retrieval score: 0.418
- strategy=hybrid
- k=1
- fixture-only: no X22 model trained or decoded

### F_hybrid (seed=1)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.851
- mean component coverage: 0.387
- leakage pass: 13
- adaptation valid: 13
- mean retrieval score: 0.418
- strategy=hybrid
- k=1
- fixture-only: no X22 model trained or decoded

### F_hybrid (seed=2)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.851
- mean component coverage: 0.387
- leakage pass: 13
- adaptation valid: 13
- mean retrieval score: 0.418
- strategy=hybrid
- k=1
- fixture-only: no X22 model trained or decoded

### G_oracle_nearest (seed=0)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.860
- mean component coverage: 0.367
- leakage pass: 13
- adaptation valid: 13
- mean retrieval score: 0.814
- strategy=oracle_nearest
- k=1
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic arm

### G_oracle_nearest (seed=1)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.860
- mean component coverage: 0.367
- leakage pass: 13
- adaptation valid: 13
- mean retrieval score: 0.814
- strategy=oracle_nearest
- k=1
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic arm

### G_oracle_nearest (seed=2)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.860
- mean component coverage: 0.367
- leakage pass: 13
- adaptation valid: 13
- mean retrieval score: 0.814
- strategy=oracle_nearest
- k=1
- fixture-only: no X22 model trained or decoded
- non-promotable diagnostic arm

### H_retrieval_as_context (seed=0)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.621
- mean component coverage: 0.453
- leakage pass: 13
- adaptation valid: 0
- mean retrieval score: 0.000
- strategy=retrieval_as_context
- k=1
- fixture-only: no X22 model trained or decoded

### H_retrieval_as_context (seed=1)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.621
- mean component coverage: 0.453
- leakage pass: 13
- adaptation valid: 0
- mean retrieval score: 0.000
- strategy=retrieval_as_context
- k=1
- fixture-only: no X22 model trained or decoded

### H_retrieval_as_context (seed=2)
- records: 13
- seed valid: 13
- mean seed-to-gold token ratio: 0.621
- mean component coverage: 0.453
- leakage pass: 13
- adaptation valid: 0
- mean retrieval score: 0.000
- strategy=retrieval_as_context
- k=1
- fixture-only: no X22 model trained or decoded

## Verdict

This is a fixture wiring run. It validates that the retrieval index is train-only, leakage-audited, and that retrieved prototypes can be hygienically remapped into hard-valid X22 initial states. Any promotable arm reporting `seed_valid_count < n_records` or a leakage failure indicates a harness bug. The oracle-nearest arm is explicitly non-promotable.
