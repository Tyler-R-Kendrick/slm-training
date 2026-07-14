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

## Round 4 — inference-speed P-series

See [perf-experiment-matrix.md](perf-experiment-matrix.md). Decode-only levers:

1. **P1 incremental grammar state** — reuse DFA + prefix text (`grammar_incremental_state`).
2. **P2 verify-chosen-only** — fewer stream probes (`grammar_verify_chosen_only`).
3. **P3 multi-token accept** — fewer denoiser forwards (`grammar_multitoken_accept`).
4. **P4 canvas lookahead** — prefix+K forwards (`grammar_canvas_lookahead`).
5. **P5 dynamic int8 quant** / compile (`use_dynamic_quant`, `use_compile`).
6. **P6 MaskGIT-primary** latency point; **P7** playground attempt budget.

## Round 5 — Q-series (admit-probe bottleneck)

After P8, `dfa_sync_ms` was ~75% of remaining latency (throwaway full-prefix
admits per candidate). Round 5:

1. **Q1** `InteractiveParser.copy()` + delta feed (`grammar_copy_probes`).
2. **Q2** whitespace fast-admit + early-exit descending-logit pick
   (`grammar_early_exit_pick`).
3. **Q9** = P8+Q1+Q2 shippable recipe (~3.2× vs P0 on CPU demo ckpt).
4. Playground defaults now enable Q9 levers; repair path ~2.3× vs pre-Q P7.

## Round 6 — R-series (exact-admit skip + repair budget)

1. **R1** skip `dfa_admits` when tid is already in an exact DFA allowed set.
2. **R2** skip redundant `set_prefix` when the engine is already synced
   (pick / force-emit / admit).
3. **R4** `_constrained_ltr_repair` honors multitoken + canvas lookahead.
4. **R5** `_ensure_valid_openui` honors `generate_max_attempts`; with
   `grammar_ltr_repair` + attempts=1, skip a redundant BOS ensure redo.
5. **R9** = Q9 + R1/R2 decode recipe; **PG** = playground with R4+R5.

```bash
python -m scripts.profile_generate --rounds 2
python -m scripts.run_perf_matrix --only P0,Q9,R9,PG --limit 4
```

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

`slm_training.runtime.accel` auto-selects **cuda → Ascend NPU → CPU**, configures thread pools,
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
Fused NEON/Cactus kernels remain external (`slm_training.runtime.cactus`).
