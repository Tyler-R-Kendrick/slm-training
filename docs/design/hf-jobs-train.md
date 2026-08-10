# Hugging Face Jobs for bounded checkpoint smoke (not ZeroGPU)

**Bounded TwoTower checkpoint smokes** run on [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-quickstart)
or multi-farm pods — **not** on Spaces [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu).

| Surface | Best for | Why |
| --- | --- | --- |
| **HF Jobs** (`scripts.hf_jobs_train`) | Managed A10G / A100 / RTX PRO checkpoint smoke | Paid flavors, canonical `slm_training.levers` timeout, `torch.compile`, bucket volumes |
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

The timeout is fixed by `slm_training.levers.HF_JOB_TIMEOUT`. Size the recipe so checkpoint sync finishes
inside that envelope; an interrupted or platform-timed-out Job is not evidence.

## Seed-fanout screening (`scripts.hf_jobs_screen`)

**Motivation (RC1,
[harness-evolution-architecture-review-20260809.md](harness-evolution-architecture-review-20260809.md)):**
local screening at n=3 smoke documents is statistically undecidable — the
minimum attainable two-sided p is 0.25, so no n=3 screening result can ever
reject at α=0.05; that is arithmetic, not an empirical finding. Properly
powered screening needs n≥8 independent seeds, which does not fit the local
`MAX_RUN_MINUTES` envelope. `scripts.hf_jobs_screen` fans one candidate config
out across `--seeds N` (default 8) HF Jobs — one seed per job — reusing this
launcher's submission machinery (`build_entrypoint_script`,
`build_jobs_run_command`, `submit_jobs_command`; `scripts.hf_jobs_train`'s CLI
is unchanged).

### Spend safety (hard requirements)

- **`--dry-run` is the default.** It prints the exact per-seed job specs,
  hardware flavor, job count, per-job timeout, and estimated wall-clock /
  GPU-minute budget — and never submits.
- Submitting requires **all three**, headlessly (there is no interactive
  prompt): `--no-dry-run`, the literal flag
  `--i-understand-this-costs-money`, and an exported `HF_TOKEN`.
- Flavor and job count are printed **before** the first submission.
- **Automation never calls this script.** No other tracked file may invoke
  `scripts.hf_jobs_screen`
  (`tests/test_scripts/test_hf_jobs_screen_policy.py` greps `git ls-files`
  to enforce it). Humans submit screening spend deliberately, by hand.

### Usage

```bash
# DEFAULT: dry-run — inspect specs, flavor, job count, budget. No spend.
python -m scripts.hf_jobs_screen --screen-id e123_screen \
  --seeds 8 --steps 200 --max-minutes 30

# Submit (opt-in spend; all three guards required)
export HF_TOKEN=hf_...
python -m scripts.hf_jobs_screen --screen-id e123_screen \
  --seeds 8 --steps 200 --max-minutes 30 \
  --no-dry-run --i-understand-this-costs-money

# Candidate config from JSON (CLI flags override file fields)
python -m scripts.hf_jobs_screen --config candidate.json --seeds 8

# Collect (no spend): fold per-seed results into one evidence JSON.
# <dir> is the bucket's screening/<screen-id>/ prefix — the volume mount
# (/mnt/openui-bucket/screening/<screen-id>) or a local `hf download` mirror.
python -m scripts.hf_jobs_screen --screen-id e123_screen --seeds 8 \
  --collect outputs/bucket_mirror/screening/e123_screen \
  --baseline outputs/screening/baseline_screen/screening_result.json
```

### Remote wall-clock and the evidence law

Each job carries a hard `--timeout {max-minutes}m` (default and hard ceiling
`MAX_RUN_MINUTES`, same as `hf_jobs_train`'s own cap) in its `hf jobs run`
spec. The per-seed result file is written only **after** train + eval + metric
extraction succeed, so a job the platform kills at the cap never publishes
one. The collector marks such seeds `"status": "incomplete"` and **excludes**
them from `values`, `sum`/`sumsq`, `mean`, and the permutation p-value — *a
timed out, interrupted, or killed run is never evidence* (same law as local
runs; `SLM_MAX_WALL_MINUTES` inside the job is set to the job's own budget).

### Result shape mapping

Per-seed jobs write `hf://buckets/…/screening/<screen-id>/seed_<k>.json`
(schema `hf_jobs_screening_seed/v1`, durable via the bucket volume mount).
`--collect` folds them into one `screening_result.json` (schema
`hf_jobs_screening/v1`) shaped for the loop's evidence consumers:

| Field | Convention it mirrors |
| --- | --- |
| `arm`, `n_obs`, `n_complete`, `sum`, `sumsq`, `mean` | per-arm accumulators (`arms.*` / `by_eval_key` `n`/`sum`/`sumsq`) in `src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json` |
| `version_stamp` (via `build_version_stamp`) | top-level stamp in `docs/design/quality-matrix-results.json` |
| `n_seeds`, `per_seed[]` (seed, run_id, status, value), `values` | screening-specific: raw per-seed evidence with explicit incomplete exclusion |
| `permutation.p_value`, `min_attainable_p`, `decidable` | `slm_training.autoresearch.power` (`min_attainable_p` / `is_decidable`) when importable; exact two-sample permutation fallback inline otherwise |

The permutation block records its `source`; the decidability floor makes the
RC1 arithmetic explicit (n=3 vs n=3 → min p 0.1, undecidable; n=8 vs n=8 →
min p ≈ 1.6e-4, decidable).

### Trackio

Each seed job mirrors its endpoint metric to Trackio after eval, following the
`slm_training.autoresearch.telemetry.TrackioSink` pattern (`trackio.init(
project=…, name=<run_id>)` + `trackio.log({metric, seed, steps})`; the
`hf` extra already installs `trackio`, and the mirror is best-effort — the
bucket JSON stays authoritative). View streams with `trackio show --project
openui-hf-screening` locally, or set the project to a Space-synced Trackio
project (`--trackio-project`) to watch remotely; job stdout also carries the
per-seed JSON (`hf jobs logs <job-id> --follow`).

## Related

- [checkpoint-bucket.md](checkpoint-bucket.md) — sync layout / auth
- [accel-parallel.md](accel-parallel.md) — AMP / compile / unmask
- [gpu-multi-farm-mcp.md](gpu-multi-farm-mcp.md) — Vast / RunPod / Lambda pods
- Hub Jobs: https://huggingface.co/docs/hub/jobs-configuration
- ZeroGPU (demos only): https://huggingface.co/docs/hub/spaces-zerogpu
