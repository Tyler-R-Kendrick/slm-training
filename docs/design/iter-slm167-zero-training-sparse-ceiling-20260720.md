# SLM-167 (SDE1-05): zero-training sparse-action ceiling fixture (slm167-zero-training-sparse-ceiling-20260720)

Matrix set: `slm167_zero_training_sparse_ceiling`

Version: `sde1-05-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

A frozen semantic scorer performs materially above random and frequency baselines on grammar-action ranking and produces some nontrivial end-to-end programs.

## Falsifier

The frozen scorer is statistically indistinguishable from strong nonsemantic baselines and produces no meaningful programs.

## Scoring arms

| arm_id | arm_name | decode_setting | seed | d_model | k_retrieve | expanded_descriptions |
| --- | --- | --- | --- | --- | --- | --- |
| random_uniform__gold_state__s0 | random_uniform | gold_state | 0 | 64 | 8 | False |
| random_uniform__free_running__s0 | random_uniform | free_running | 0 | 64 | 8 | False |
| global_frequency__gold_state__s0 | global_frequency | gold_state | 0 | 64 | 8 | False |
| global_frequency__free_running__s0 | global_frequency | free_running | 0 | 64 | 8 | False |
| compiler_local_frequency__gold_state__s0 | compiler_local_frequency | gold_state | 0 | 64 | 8 | False |
| compiler_local_frequency__free_running__s0 | compiler_local_frequency | free_running | 0 | 64 | 8 | False |
| permuted_descriptions__gold_state__s0 | permuted_descriptions | gold_state | 0 | 64 | 8 | False |
| permuted_descriptions__free_running__s0 | permuted_descriptions | free_running | 0 | 64 | 8 | False |
| bi_encoder_similarity__gold_state__s0 | bi_encoder_similarity | gold_state | 0 | 64 | 8 | False |
| bi_encoder_similarity__free_running__s0 | bi_encoder_similarity | free_running | 0 | 64 | 8 | False |
| frozen_continuation__gold_state__s0 | frozen_continuation | gold_state | 0 | 64 | 8 | False |
| frozen_continuation__free_running__s0 | frozen_continuation | free_running | 0 | 64 | 8 | False |
| hybrid_retrieval_rerank__gold_state__s0 | hybrid_retrieval_rerank | gold_state | 0 | 64 | 8 | False |
| hybrid_retrieval_rerank__free_running__s0 | hybrid_retrieval_rerank | free_running | 0 | 64 | 8 | False |
| small_model_control__gold_state__s0 | small_model_control | gold_state | 0 | 64 | 8 | False |
| small_model_control__free_running__s0 | small_model_control | free_running | 0 | 64 | 8 | False |

## Results

| arm_id | arm_name | decode_setting | seed | top1 | top3 | top5 | MRR | NDCG@5 | meaningful_program_rate | rare_recall | parse_validity | full_set_recall | wall_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_uniform__gold_state__s0 | random_uniform | gold_state | 0 | 0.167 | 0.333 | 0.417 | 0.313 | 0.297 | 0.105 | 0.078 | 0.144 | 1.000 | 0.072 |
| random_uniform__free_running__s0 | random_uniform | free_running | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.087 | 0.068 | 0.137 | 1.000 | 0.058 |
| global_frequency__gold_state__s0 | global_frequency | gold_state | 0 | 0.083 | 0.167 | 0.333 | 0.210 | 0.189 | 0.160 | 0.147 | 0.229 | 1.000 | 0.059 |
| global_frequency__free_running__s0 | global_frequency | free_running | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.131 | 0.128 | 0.164 | 1.000 | 0.070 |
| compiler_local_frequency__gold_state__s0 | compiler_local_frequency | gold_state | 0 | 0.083 | 0.250 | 0.500 | 0.248 | 0.278 | 0.180 | 0.157 | 0.235 | 1.000 | 0.051 |
| compiler_local_frequency__free_running__s0 | compiler_local_frequency | free_running | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.151 | 0.142 | 0.202 | 1.000 | 0.070 |
| permuted_descriptions__gold_state__s0 | permuted_descriptions | gold_state | 0 | 0.083 | 0.500 | 0.750 | 0.345 | 0.428 | 0.116 | 0.109 | 0.173 | 1.000 | 0.082 |
| permuted_descriptions__free_running__s0 | permuted_descriptions | free_running | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.096 | 0.087 | 0.137 | 1.000 | 0.056 |
| bi_encoder_similarity__gold_state__s0 | bi_encoder_similarity | gold_state | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.270 | 0.263 | 0.322 | 1.000 | 0.073 |
| bi_encoder_similarity__free_running__s0 | bi_encoder_similarity | free_running | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.220 | 0.199 | 0.276 | 1.000 | 0.053 |
| frozen_continuation__gold_state__s0 | frozen_continuation | gold_state | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.296 | 0.293 | 0.346 | 1.000 | 0.052 |
| frozen_continuation__free_running__s0 | frozen_continuation | free_running | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.246 | 0.226 | 0.284 | 1.000 | 0.051 |
| hybrid_retrieval_rerank__gold_state__s0 | hybrid_retrieval_rerank | gold_state | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.315 | 0.311 | 0.352 | 0.972 | 0.058 |
| hybrid_retrieval_rerank__free_running__s0 | hybrid_retrieval_rerank | free_running | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.284 | 0.247 | 0.348 | 0.965 | 0.067 |
| small_model_control__gold_state__s0 | small_model_control | gold_state | 0 | 0.833 | 1.000 | 1.000 | 0.917 | 0.938 | 0.395 | 0.373 | 0.464 | 1.000 | 0.065 |
| small_model_control__free_running__s0 | small_model_control | free_running | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.342 | 0.326 | 0.377 | 1.000 | 0.063 |

## Per-arm means

| arm_name | top1 | top3 | top5 | MRR | NDCG@5 | meaningful_program_rate | rare_recall | parse_validity | full_set_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_uniform | 0.083 | 0.167 | 0.208 | 0.157 | 0.148 | 0.096 | 0.073 | 0.141 | 1.000 |
| global_frequency | 0.042 | 0.083 | 0.167 | 0.105 | 0.095 | 0.146 | 0.137 | 0.197 | 1.000 |
| compiler_local_frequency | 0.042 | 0.125 | 0.250 | 0.124 | 0.139 | 0.166 | 0.150 | 0.219 | 1.000 |
| permuted_descriptions | 0.042 | 0.250 | 0.375 | 0.173 | 0.214 | 0.106 | 0.098 | 0.155 | 1.000 |
| bi_encoder_similarity | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.245 | 0.231 | 0.299 | 1.000 |
| frozen_continuation | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.271 | 0.259 | 0.315 | 1.000 |
| hybrid_retrieval_rerank | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.300 | 0.279 | 0.350 | 0.968 |
| small_model_control | 0.417 | 0.500 | 0.500 | 0.458 | 0.469 | 0.368 | 0.349 | 0.421 | 1.000 |

## Disposition

**useful_zero_training_prior**

Frozen semantic scoring materially exceeds nonsemantic baselines and produces a nontrivial meaningful-program rate, indicating a usable zero-training prior.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The frozen scorer, baselines, and metrics are exercised over deterministic synthetic inputs, but no real model was trained or evaluated. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained scorer and AgentV evaluation are available.

## Honest caveats

- Metrics are generated by a deterministic simulator, not a trained model.
- The frozen continuation scorer is a local hash-based proxy; real causal-LM   scoring belongs to SLM-108.
- Bi-encoder similarity uses the SLM-163 hash-based fixture encoder, not a   pretrained sentence transformer.
- No content floor, prompt inventory, hidden slot contract, or retry is used.
- Free-running generation is simulated; real compiler transition errors are   not exercised here.
- No Pareto or ship-gate claim is made; this is wiring evidence only.

## Reproducibility

```bash
python -m scripts.run_slm167_zero_training_sparse_ceiling_fixture --mode plan-only
python -m scripts.run_slm167_zero_training_sparse_ceiling_fixture --mode fixture
```
