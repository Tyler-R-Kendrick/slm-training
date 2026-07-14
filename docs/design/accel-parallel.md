# Accelerator & parallel decode notes

## Environment

This cloud agent run: **CPU-only** (no CUDA / Ascend NPU). The accel layer still
auto-selects `cuda → npu → cpu` so the same CLI works on GPU/NPU hosts.

## Techniques applied

Canonical citations and fidelity tags: [research-lineage.md](research-lineage.md).

| Technique | Source | Integration |
| --- | --- | --- |
| SDPA attention | PyTorch `scaled_dot_product_attention` | Already in `blocks.MultiheadAttention` |
| Adaptive parallel unmask | MaskGIT + dLLM-style confidence / spacing (**Adapted**) | `models/parallel_decode.py` → MaskGIT |
| Grammar force-emit / admit | Mündler et al. 2025 + Leviathan-adjacent (**Adapted**) | `grammar_fastpath/` · [grammar-fastpath.md](grammar-fastpath.md) |
| torch.compile + CUDA graphs | PyTorch 2.x serving guides | `accel.maybe_compile` (GPU); CPU skipped without Python.h |
| AMP bf16/fp16 | Standard accelerator train | `accel.autocast_context` + `--amp` |
| Grad accumulation | Effective batch without OOM | `--grad-accum` |
| Threaded grammar checks | Overlap Node bridge latency | `_repair_ltr_texts` ThreadPool |
| Matrix workers | Parallel independent experiments | `run_quality_matrix --workers N` |
| BF16 exponent codebook | [brianbell-x weight-compression](https://brianbell-x.github.io/weight-compression/) | Storage sidecar; fused kernel external |

## Bench (this host)

See `outputs/runs/accel_bench.json` — ~11.5 train steps/s, ~39 generate prompts/s
on 4-vCPU for d_model=128 LTR-primary. Adaptive unmask ≈ topk latency (quality knob).

## Train microbench

`scripts/bench_accel.py --microbench` compares cache/fuse variants and writes
`docs/design/train-microbench.json`. Winning defaults: `cache_context=True`,
`fuse_ltr_loss=True`, CLI `--fast-train` (AMP when accel supports it + compile).

**Full GPU trains:** use [HF Jobs](hf-jobs-train.md) (`scripts.hf_jobs_train`) or
pods (`scripts.remote_train`), not Spaces ZeroGPU. Jobs/pods get TF32,
cudnn.benchmark, `--fast-train`, and `--compile-mode reduce-overhead`. Auto
`SLM_FAST_TRAIN` / `HF_JOB_ID` enable the speed bundle; ZeroGPU never does.

Rejected for this tiny stack: QLoRA/Unsloth (denoiser ~1.5M), Cactus-inside-train,
MoE, per-step linter.

## E9 follow-up

`qx_e9_accel_combo`: capacity + curriculum + fidelity + schema + retrieval +
adaptive unmask + LTR repair, 1000 steps, grad-accum 2.
