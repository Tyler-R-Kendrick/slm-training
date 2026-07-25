# SLM-216: SpectralRegimeGateV1 (slm216-spectral-regime-20260723)

**Status / claim:** `scratch_measured` / `diagnostic` (`scratch_cpu`)

**Verdict:** `inconclusive`

**Report hash:** `7fd9f53499195a196080a24748451ced1c5eea89fb52c3a1519e2f6ae1e88675`

**Semantic floor:** `7839ef6b6e37710d487757da9170017d7b76a9d12ca1fb314bdb0fa23a4dd83d` (`inconclusive`)

**Source commit / matrix manifest:** `f70a46e12be5271e217c68f176ba455eb2deaf0e` / `fa24671bd6a8d6b67329baa29e1dd3382df56810341610f1779a0afa501248c3`

**Recipe:** CPU scratch `TinySpectralModel/16x16/v1`; AdamW, LR 0.02, weight decay 0; 1,280 target tokens per primary cell; snapshots at 0/640/1,280 tokens; five same-shape Kaiming-null draws; three seeds.

## Preregistered matrix

| Cell | Seed | Physical / accumulation / effective batch | Data | Tokens | Steps | Unique / repeated | Final ESD distance | Held-out MSE |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: |
| `batch2_scale1` | 0 | 2 / 1 / 2 | 1× diverse | 1280 | 80 | 16 / 144 | 0.327293 | 0.188272 |
| `batch2_scale1` | 1 | 2 / 1 / 2 | 1× diverse | 1280 | 80 | 16 / 144 | 0.367978 | 0.248964 |
| `batch2_scale1` | 2 | 2 / 1 / 2 | 1× diverse | 1280 | 80 | 16 / 144 | 0.311792 | 0.388015 |
| `batch8_scale1` | 0 | 8 / 1 / 8 | 1× diverse | 1280 | 20 | 16 / 144 | 0.303470 | 0.404685 |
| `batch8_scale1` | 1 | 8 / 1 / 8 | 1× diverse | 1280 | 20 | 16 / 144 | 0.333848 | 0.459641 |
| `batch8_scale1` | 2 | 8 / 1 / 8 | 1× diverse | 1280 | 20 | 16 / 144 | 0.330827 | 0.648328 |
| `physical2_accum4_scale1` | 0 | 2 / 4 / 8 | 1× diverse | 1280 | 20 | 16 / 144 | 0.303470 | 0.404685 |
| `physical2_accum4_scale1` | 1 | 2 / 4 / 8 | 1× diverse | 1280 | 20 | 16 / 144 | 0.333848 | 0.459641 |
| `physical2_accum4_scale1` | 2 | 2 / 4 / 8 | 1× diverse | 1280 | 20 | 16 / 144 | 0.330827 | 0.648328 |
| `batch8_scale5` | 0 | 8 / 1 / 8 | 5× diverse | 1280 | 20 | 80 / 80 | 0.221429 | 0.223936 |
| `batch8_scale5` | 1 | 8 / 1 / 8 | 5× diverse | 1280 | 20 | 80 / 80 | 0.232553 | 0.256271 |
| `batch8_scale5` | 2 | 8 / 1 / 8 | 5× diverse | 1280 | 20 | 80 / 80 | 0.186706 | 0.286033 |
| `batch8_scale10` | 0 | 8 / 1 / 8 | 10× diverse | 1280 | 20 | 160 / 0 | 0.225855 | 0.197720 |
| `batch8_scale10` | 1 | 8 / 1 / 8 | 10× diverse | 1280 | 20 | 160 / 0 | 0.219946 | 0.222842 |
| `batch8_scale10` | 2 | 8 / 1 / 8 | 10× diverse | 1280 | 20 | 160 / 0 | 0.184614 | 0.247945 |
| `batch8_scale5_duplicated` | 0 | 8 / 1 / 8 | 5× duplicated | 1280 | 20 | 16 / 144 | 0.303470 | 0.404685 |
| `batch8_scale5_duplicated` | 1 | 8 / 1 / 8 | 5× duplicated | 1280 | 20 | 16 / 144 | 0.333848 | 0.459641 |
| `batch8_scale5_duplicated` | 2 | 8 / 1 / 8 | 5× duplicated | 1280 | 20 | 16 / 144 | 0.330827 | 0.648328 |

## Paired scratch effects

| Contrast (right − left final ESD distance) | Paired seeds | Mean delta | Standard error |
| --- | ---: | ---: | ---: |
| Batch 8 − batch 2 (fixed tokens; step-confounded) | 3 | -0.012973 | 0.016278 |
| Diverse 10× − diverse 1× | 3 | -0.112576 | 0.019814 |
| Diverse 5× − duplicated 5× | 3 | -0.109152 | 0.018347 |

Scratch outcome relationship: `pearson_r_final_esd_vs_heldout_mse` = `0.588157` across `n=18` final cells. This descriptive association is not current-model or causal evidence.

Direct physical batch 8 and physical batch 2 with accumulation 4 produce identical state hashes and trajectories at matched effective batch/optimizer steps. This verifies accumulation accounting; it does not identify an effective-batch causal effect independently of step count.

## Gate rationale

- no durable current-model checkpoint references
- SemanticFloorGateV1 is inconclusive; semantic spectral claims are blocked
- the executed cells are deterministic CPU scratch controls, not the current serving model/data regime
- the fixed-token batch-2/batch-8 contrast changes optimizer-step count; the direct-batch/accumulated-batch control isolates physical batching only

## Allowed downstream

- `spectral_diagnostics`
- `scratch_harness_validation`

## Blocked claims

- `spectral_lr_control`
- `spectral_rg_control`
- `semantic_prediction`
- `semantic_causal`
- `promotion`
- `ship`

## Reproduction

```bash
timeout 170s env PYTHONPATH=src .venv/bin/python -m scripts.run_spectral_regime_matrix --check
```

No reusable checkpoint was written or promoted. This measured scratch result does not establish that the current serving-model regime is spectrally addressable.
No canonical model evaluation or AgentV run was performed; held-out scratch MSE is a wiring diagnostic, not a ship metric.
