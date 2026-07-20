# SLM-133 — AST-sketch dedup × choice-native retrieval factorial (wiring)

## What

Preregistered, corrected EFS3-06 factorial manifest for testing two
data-efficiency hypotheses together:

1. AST-sketch balancing improves semantic coverage at fixed corpus/exposure budget.
2. Representation-aligned (choice-native) retrieval improves semantic selection
   over no retrieval and over random/surface-exemplar controls.

This is the wiring/fixture slice. It defines the schemas, retriever interface,
and preregistered matrix; it does not train models or make ship claims.

## Matrix registration

- `matrix_set`: `ast-sketch-retrieval`
- `matrix_version`: `efs3-06-v1`
- `experiment_id`: `efs-ast-sketch-retrieval`

## Arms

| Arm | Data sampling | Retrieval mode | Seeds | K | Context budget |
| --- | --- | --- | --- | --- | --- |
| raw_stratified__none | raw_stratified | none | 0,1,2 | 4 | 400 |
| ast_sketch_balanced__none | ast_sketch_balanced | none | 0,1,2 | 4 | 400 |
| raw_stratified__choice_exemplar | raw_stratified | choice_exemplar | 0,1,2 | 4 | 400 |
| ast_sketch_balanced__choice_exemplar | ast_sketch_balanced | choice_exemplar | 0,1,2 | 4 | 400 |
| raw_stratified__random_choice | raw_stratified | random_choice | 0,1,2 | 4 | 400 |
| raw_stratified__surface_skeleton | raw_stratified | surface_skeleton | 0,1,2 | 4 | 400 |

Row set = **6 arms × 3 seeds = 18 primary/control rows**.

## Frozen base recipe

The recipe is the E228 legal-candidate-margin recipe extended with choice-native
decoding, `d_model=128`, `retrieval_k=4`, and `retrieval_context_budget=400`.
Its SHA-256 is stored in the manifest.

## Files added

- `src/slm_training/harnesses/experiments/ast_sketch_retrieval_factorial.py`
- `scripts/run_ast_sketch_retrieval_factorial.py`
- `tests/test_harnesses/experiments/test_ast_sketch_retrieval_factorial.py`
- `tests/test_scripts/test_run_ast_sketch_retrieval_factorial.py`
- `docs/design/iter-efs3-06-ast-sketch-retrieval-factorial-20260719.md`
- `docs/design/iter-efs3-06-ast-sketch-retrieval-factorial-20260719.json`

## Commands

```bash
# Plan only (CPU, no model load)
python -m scripts.run_ast_sketch_retrieval_factorial --mode plan-only \
  --output-dir outputs/runs/slm133_ast_sketch_retrieval

# Fixture wiring check (includes controls)
python -m scripts.run_ast_sketch_retrieval_factorial --mode fixture \
  --include-controls \
  --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
  --output-dir outputs/runs/slm133_ast_sketch_retrieval_fixture
```

## Verification

- `pytest tests/test_harnesses/experiments/test_ast_sketch_retrieval_factorial.py -q` → 22 passed
- `pytest tests/test_scripts/test_run_ast_sketch_retrieval_factorial.py -q` → 3 passed
- `python -m scripts.verify_version_stamps --check` → ok
- `python -m scripts.repo_policy` → ok

## Honest caveats

This is **wiring evidence only**. The actual factorial requires a labeled
semantic corpus (SLM-105), the EFS1 exposure decision (SLM-109), a trained
choice-native checkpoint (SLM-124), and GPU hosts. The `frontier` mode emits a
fixture plan and a clear stderr notice. No data-efficiency or retrieval claim
is made from this artifact.
