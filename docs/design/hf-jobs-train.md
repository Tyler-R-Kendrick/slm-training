# Hugging Face Jobs for bounded checkpoint smoke (not ZeroGPU)

**Bounded TwoTower checkpoint smokes** run on [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-quickstart)
or multi-farm pods — **not** on Spaces [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu).

| Surface | Best for | Why |
| --- | --- | --- |
| **HF Jobs** (`scripts.hf_jobs_train`) | Managed A10G / A100 / RTX PRO checkpoint smoke | Paid flavors, hard three-minute timeout, `torch.compile`, bucket volumes |
| **Pods** (`scripts.remote_train` + multi-farm MCP) | Bring-your-own GPU / cheapest spot | Same `--fast-train` knobs over SSH |
| **ZeroGPU** Gradio Spaces | Short **demo inference** only | `@spaces.GPU` minutes of quota; **no** `torch.compile`; process isolation |

## Why not ZeroGPU for training?

- Daily GPU quotas are minutes (PRO ~40m included), not multi-hour ship runs
- `torch.compile` / Inductor CUDA graphs are unsupported (AoTI is for demos)
- Gradio-only; our train path needs Node grammar bridges, long steps, bucket sync
- Workers are forked per request and killed when slots recycle

Use ZeroGPU only if you later ship a Gradio playground that loads a **synced**
checkpoint under `@spaces.GPU` (no compile; AoTI optional). Training itself
always goes through Jobs or pods.

## Submit a Job

Prerequisites: Hub Pro/Team/Enterprise credits, write `HF_TOKEN`, `hf` CLI
(`hf auth login`).

```bash
# Preview command + entrypoint (no submit)
python -m scripts.hf_jobs_train --dry-run \
  --run-id twotower_jobs_v1 --steps 200 --branch main

# Submit (A10G large, hard 3m timeout, mounts checkpoint bucket)
export HF_TOKEN=hf_...
python -m scripts.hf_jobs_train \
  --flavor a10g-large \
  --timeout 3m \
  --run-id twotower_jobs_v1 \
  --steps 200 \
  --branch main
```

Equivalent raw CLI shape (built by the launcher):

```bash
hf jobs run \
  --flavor a10g-large \
  --timeout 3m \
  --secrets HF_TOKEN \
  --env SLM_FAST_TRAIN=1 \
  --volume hf://buckets/TKendrick/OpenUI:/mnt/openui-bucket \
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  bash -lc '<clone repo + pip/npm + train_model --fast-train …>'
```

Monitor:

```bash
hf jobs ps
hf jobs logs <job-id> --follow
hf jobs cancel <job-id>
hf jobs hardware   # flavors + rates
```

## Performance knobs (applied automatically)

| Knob | Where |
| --- | --- |
| TF32 + cudnn.benchmark + expandable allocator | `accel.configure_cuda_training()` via `detect_device` |
| `--fast-train` (cache context, fuse LTR, AMP, compile) | Jobs env `SLM_FAST_TRAIN` / `HF_JOB_ID`; CLI on pods |
| `--compile-mode reduce-overhead` (CUDA graphs) | Jobs + `remote_train` defaults on CUDA |
| Checkpoint sync to `hf://buckets/TKendrick/OpenUI` | `--sync-checkpoints` (HF context default) |
| Bucket volume mount | `--volume hf://buckets/…` (Jobs); durable even if Job dies |

Disable speed bundle: `--no-fast-train` or `SLM_FAST_TRAIN=0`.

ZeroGPU Spaces never auto-enable `--fast-train` (`accel.is_zerogpu_environment`).

## Flavors

Pick with `hf jobs hardware`. Common choices:

- `a10g-large` — default for this repo’s small TwoTower footprint
- `a100-large` variants — maximize work completed inside the fixed run cap
- `rtx-pro-6000` family — when available on Jobs (same generation as ZeroGPU backing)

The timeout is fixed at three minutes. Size the recipe so checkpoint sync finishes
inside that envelope; an interrupted or platform-timed-out Job is not evidence.

## Related

- [checkpoint-bucket.md](checkpoint-bucket.md) — sync layout / auth
- [accel-parallel.md](accel-parallel.md) — AMP / compile / unmask
- [gpu-multi-farm-mcp.md](gpu-multi-farm-mcp.md) — Vast / RunPod / Lambda pods
- Hub Jobs: https://huggingface.co/docs/hub/jobs-configuration
- ZeroGPU (demos only): https://huggingface.co/docs/hub/spaces-zerogpu
