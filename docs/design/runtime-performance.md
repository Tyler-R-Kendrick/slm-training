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

## Playground load reproduction (2026-07-15)

The [runtime reproduction](playground-runtime-reproduction-results.json) found
three independent failures in the React playground path. A fresh worktree could
throw a bridge-install error during validation, a `web`-only install could fail
to import PyTorch despite committed ONNX artifacts, and the SPA multiplied its
own six retries by the service's three-attempt budget while prefetching three
samples (up to 54 server decodes). One PyTorch attempt took roughly 6–9 seconds
on this CPU fixture host, so the amplification looked like a hung application.

The web service now validates through the hybrid parser and uses committed ONNX
artifacts only when CPU PyTorch is unavailable. The React flow performs exactly
three numbered server attempts, then at most three browser attempts with failure
context and durable attempt/review records; navigating away aborts the active
pipeline and destroys any late browser session. Browser startup and inference are
bounded, WebGPU adapters below the model's 65,536-byte workgroup-storage
requirement are skipped, and an explicitly labeled wiring fallback keeps the
annotation/editor flow usable when neither model backend works. That fallback is
excluded from derived training data unless a human corrects it.

The production dashboard build passed, the desktop Playwright suite passed 12/12
in 58.5 seconds, the mobile suite passed 12/12 in 1.0 minute, and 16 focused
backend tests passed in 0.67 seconds. Four deterministic desktop parity workflows
also passed in 12.2 seconds. A real Windows Chrome run against the committed CPU
checkpoint reached a valid rendered sample in 15,378 ms, enabled grading, and
saved a renderer-validated human correction through `/api/annotate`. It recorded
zero page errors, console errors, failed requests, or HTTP errors. In that run the
checkpoint's three attempts produced invalid DSL, WebGPU was correctly rejected
at 32,768 < 65,536 bytes, and ONNX WASM rejected `GatherBlockQuantized(1)`; the
non-training wiring fallback then rendered successfully. This is fixture-demo
runtime evidence, not an eval or ship-readiness claim; no checkpoint, suite, or
ship gate changed.

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

`slm_training.runtime.accel` auto-selects **CUDA → Ascend NPU → DirectML → CPU**, configures
thread pools, and exposes AMP + `torch.compile` where the backend supports them.

| Knob | Flag | Notes |
| --- | --- | --- |
| Device | `--device auto` | CUDA, Ascend NPU, or Torch-DirectML when present |
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

### Exact choice-state cache (E289, 2026-07-17)

Choice-native constrained decoding now memoizes exact legal-token sets by
immutable production-state signature and remaining positions. On the same
checkpoint bytes as E288, all five suites retained parse 1.0 with zero dead
ends. Standalone p50 latency improved 2.65×–5.86×; cache hit rates were
57.6%–76.4%. Cold-state p95 remained about 5.9–8.7 seconds, so direct
pushdown-frame candidate construction is the next target. Semantic gates remain
zero and AgentV is 0/5; this is a runtime improvement, not a ship result.
See [E289 results](iter-e289-choice-state-cache-20260717.md).

### Grammar-derived choice candidates (E290, 2026-07-17)

The choice decoder now constructs a next-token superset from production/frame
categories before applying the unchanged exact validator. It avoids 34.8% of
whole-vocabulary probes on cache misses and preserves all-suite parse 1.0 with
zero dead ends. Across two standalone runs, p95 improves 14–19% over E289, but
p50 regresses 11–41%; therefore this is a mixed result, not a promoted runtime
default. See [E290 results](iter-e290-choice-direct-candidates-20260717.md).

### Exact completion-state cache (E291, 2026-07-17)

Minimum completion lengths are now memoized by immutable choice-decoder state,
and expression partitions are reused by slot/reference counts. A schema-warm
control improves exact allowed-set construction 7.0×. Across two all-suite
runs, completion-cache hit rates are 90.7–91.9%; p50 improves 1.29×–1.99× and
p95 1.51×–1.93× over E290 with identical outputs and zero dead ends. Lazy
component-contract construction still adds about 2.38 seconds to the first
process-cold request. See [E291 results](iter-e291-choice-completion-cache-20260717.md).

### Restricted semantic projection

TwoTower's denoiser now exposes a decode-only `encode` / `project` split. The
legacy forward still computes the full tied vocabulary head, while compiler
decode gathers only valid semantic rows. Packed trie verification batches
prefix-visible/future-masked canvases and projects each parent's child set.
This removes LM-head work, not backbone work; deterministic singleton spans
are the only path that skips neural inference entirely. See the C-series in
[`perf-experiment-matrix.md`](perf-experiment-matrix.md). The feature remains
opt-in because the current committed checkpoint cannot provide a valid quality
anchor.

### Local Qualcomm DirectML train (2026-07-14)

[`local_directml_adreno_20260714`](local-directml-train-results.json) completed a real
five-step TwoTower scratch train on the Windows Qualcomm Adreno X1-85 GPU through
Torch-DirectML (`privateuseone:0`). The 924,386-parameter run processed 5,120 prompt
and 3,581 target tokens, reported 4,861.289 ms total cycle telemetry, wrote a
3,727,242-byte checkpoint, and reloaded it on CPU. This is accelerator/checkpoint
wiring evidence only: no eval suite or ship gates ran. DirectML moved the unsupported
AdamW `aten::lerp.Scalar_out` operator to CPU, so the run is GPU-backed rather than
pure-GPU. The checkpoint loaded and passed the CPU playground health check, but a
real generation did not return within 120 seconds; the known-good fixture remains
the local demo checkpoint.
