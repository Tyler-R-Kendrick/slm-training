# Runtime performance notes

Measured on `twotower_v1_ship` (CPU, scratch context, LTR primary).

## Hotspots

| Path | Before (initial) | After round 1 | After round 2 |
| --- | --- | --- | --- |
| Single `generate` | ~111ms | ~71ms | **~63ms** |
| Eval 64× RICO (sequential) | — | — | ~4070ms |
| Eval 64× RICO (`generate_batch` 16) | — | — | **~1005ms (~4×)** |
| DESIGN.md lint (repeat) | ~75ms | ~0.005ms | same |
| Eval gold design lint | ~75ms×N | ~0 (meta) | same |
| Cactus / NEON kernel | n/a | separate | separate |

## Round 2 changes

1. **Batched LTR decode** (`generate_batch`) with progressive canvases; eval uses it automatically.
2. **Active-row subsetting** — finished EOS rows drop out of the transformer forward.
3. **Stages `(64, 128, 192, 256)`** (current `grammar_ltr_stages` default; earlier notes used `(32, 48, 96)`) — shorter programs exit earlier.
4. **Skip Node finalize validate on generate** (eval already validates); optional via `grammar_finalize_validate`.
5. **OpenUI parse/validate result cache** in `lang_core`.

## Round 3 — train speed + grammar fast-path

1. **Frozen HF backbone + DESIGN.md string cache** (`cache_context`; `--fast-train`).
2. **Fused mask+LTR** into one denoiser forward (`fuse_ltr_loss`).
3. **Vectorized** mask/pad; fidelity via `isin`.
4. **Grammar force-emit / MaskGIT admit** (`grammar_fastpath`); Cactus kernel sketches under `cactus/kernels/`.
5. Microbench: `scripts/bench_accel.py --microbench` → `docs/design/train-microbench.json`
   (scratch fuse+cache ~1.7× vs baseline on this host; HF cache is the production win).

## Kernel boundary (unchanged)

```
prompt → TwoTower (PyTorch) → OpenUI text
                ↓ export_checkpoint_bundle
         portable .pt + tokenizer  →  (offline) cactus-compute → .cact / NEON
                              ↘ optional model.pt.wc.json (BF16 codebook)
```

Do not vendor Cactus NEON kernels into `slm_training.models`.

## Lossless weight compression

Reference: [brianbell-x weight-compression](https://brianbell-x.github.io/weight-compression/)
(candidate 0009 fusible exponent codebook).

| Piece | Choice |
| --- | --- |
| View | FP32 checkpoint → BF16 bits (lossless vs BF16, not raw FP32) |
| Layout | `regroup` (~11.3 b/w headline) or `bytesplit` (GPU-validated) |
| Index | Top-K=15 sign+exponent symbols → 4-bit code + escape |
| Payload | 7-bit mantissa (regroup) / low byte (bytesplit) + in-order escapes |
| Runtime | PyTorch path **decompresses to float32**; fused read/matmul kernels stay external |

```bash
python -m scripts.compress_weights \
  --checkpoint outputs/runs/twotower_v1_ship/checkpoints/last.pt \
  --layout regroup --verify-load
```

`export_checkpoint_bundle(..., compress_weights=True)` writes `model.pt.wc.json` +
stats sidecar. On a TwoTower-scale synthetic FP32 state (~1.9M weights):

| Layout | bits/weight | vs BF16 | bit-exact BF16 |
| --- | --- | --- | --- |
| `regroup_k15` | ~11.17 | **~30.2%** | yes |
| `bytesplit_k15` | ~12.01 | ~25.0% | yes |

The JSON sidecar stores hex streams for portability; `compressed_bytes` in the
stats file is the packed fusible bit budget (what a fused engine would keep in
VRAM). Compression is storage/VRAM-oriented; it does not change the generate hot
path until an external fused engine consumes the narrow form.

## Accelerator utilization

`slm_training.accel` auto-selects **cuda → Ascend NPU → CPU**, configures thread pools,
and exposes AMP + `torch.compile` for train/decode.

| Knob | Flag | Notes |
| --- | --- | --- |
| Device | `--device auto` | CUDA/NPU when present |
| AMP | `--amp` | bf16/fp16 autocast on accelerators |
| Compile | `--compile` | Inductor; `reduce-overhead` → CUDA graphs on GPU |
| Grad accum | `--grad-accum N` | Larger effective batch without OOM |
| Parallel unmask | `--parallel-unmask adaptive` | MaskGIT + confidence/spacing (**Adapted**; [research-lineage](research-lineage.md)) |
| Matrix workers | `--workers 2` | Parallel independent train experiments |

```bash
python -m scripts.bench_accel --device auto
python -m scripts.train_model --device auto --compile --amp --grad-accum 2 \
  --parallel-unmask adaptive --grammar-ltr-primary --steps 800
```

This environment's measured accel bench is recorded under `outputs/runs/accel_bench.json`.
Fused NEON/Cactus kernels remain external (`slm_training.cactus`).
