# Accelerator & parallel decode notes

## Environment

This cloud agent run: **CPU-only** (no CUDA / Ascend NPU). The accel layer still
auto-selects `cuda → npu → cpu` so the same CLI works on GPU/NPU hosts.

## SOTA techniques applied

| Technique | Source | Integration |
| --- | --- | --- |
| SDPA attention | PyTorch scaled_dot_product_attention | Already in `blocks.MultiheadAttention` |
| Adaptive parallel unmask | 2026 dLLM mean-field / confidence schedules | `models/parallel_decode.py` → MaskGIT |
| torch.compile + CUDA graphs | PyTorch 2.x / 2026 serving guides | `accel.maybe_compile` (GPU); CPU skipped without Python.h |
| AMP bf16/fp16 | Standard accelerator train | `accel.autocast_context` + `--amp` |
| Grad accumulation | Effective batch without OOM | `--grad-accum` |
| Threaded grammar checks | Overlap Node bridge latency | `_repair_ltr_texts` ThreadPool |
| Matrix workers | Parallel independent experiments | `run_quality_matrix --workers N` |
| BF16 exponent codebook | brianbell-x weight-compression | Storage sidecar; fused kernel external |

## Bench (this host)

See `outputs/runs/accel_bench.json` — ~11.5 train steps/s, ~39 generate prompts/s
on 4-vCPU for d_model=128 LTR-primary. Adaptive unmask ≈ topk latency (quality knob).

## E9 follow-up

`qx_e9_accel_combo`: capacity + curriculum + fidelity + schema + retrieval +
adaptive unmask + LTR repair, 1000 steps, grad-accum 2.
