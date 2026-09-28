# Hugging Face Jobs Execution

Run training on Hugging Face's managed GPUs without provisioning any local infrastructure. The same training script runs locally and on Jobs. This reference covers only the Jobs-specific concerns.

## Prerequisites

- Hugging Face account with a **Pro, Team, or Enterprise** plan. Jobs are paid.
- `HF_TOKEN` with **write** permission. Log in once locally with `hf auth login` (the modern command from the `hf` CLI, replacing the deprecated `huggingface-cli login`).
- Access to the `hf_jobs()` MCP tool, or the `hf` CLI (`curl -LsSf https://hf.co/cli/install.sh | bash -s`).

## The three submission paths

### 1. Inline script via MCP (recommended in Claude Code)

Pass the full training script as `script`. Dependencies come from the PEP 723 header.

```python
hf_jobs("uv", {
    "script": """
# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers[train]>=5.0", "trackio"]
# ///

# <full training script content>
""",
    "flavor": "a10g-large",
    "timeout": "3h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"},
})
```

### 2. Script-from-URL via MCP

Upload the script to the Hub (as a model or dataset repo file) or a Gist, then reference by URL:

```python
hf_jobs("uv", {
    "script": "https://huggingface.co/USERNAME/scripts/resolve/main/train_bi_encoder.py",
    "flavor": "a10g-large",
    "timeout": "3h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"},
})
```

Local file paths (`./train.py`, `/path/to/train.py`) **do not work**. Jobs run in isolated containers without access to your filesystem.

### 3. CLI

```bash
hf jobs uv run \
    --flavor a10g-large \
    --timeout 3h \
    --secrets HF_TOKEN \
    "https://huggingface.co/USERNAME/scripts/resolve/main/train.py"
```

Syntax gotchas:
- Command order is `hf jobs uv run`, **not** `hf jobs run uv`.
- Flags (`--flavor`, `--timeout`, `--secrets`) go **before** the script URL.
- `--secrets` (plural), not `--secret`.

## Durable checkpoint storage for Jobs

Choose an approved persistent destination before submission. In this repository,
follow `AGENTS.md` checkpoint-bucket policy. Hub publication is optional and
requires explicit authorization for the target repository.

Add these to your `TrainingArguments`:

```python
args = SentenceTransformerTrainingArguments(
    ...,
    push_to_hub=False,                 # enable only after explicit authorization
    save_strategy="steps",
    save_steps=0.1,                   # 10 saves/pushes per epoch; scales with dataset size
)
```

Why each matters:

| Argument | Why |
|---|---|
| Approved persistent storage | Required when artifacts must survive container shutdown. Configure repository-approved bucket or persistent volume, then verify receipt. |
| `push_to_hub=True` | Optional. Enable only after explicit authorization for the exact destination. |
| `hub_model_id` | Required only when authorized Hub upload is enabled. |
| `hub_strategy="every_save"` | Default, but worth being deliberate about on Jobs: each checkpoint is pushed as it's written, so a timeout leaves all completed checkpoints on the Hub. `"end"` only pushes once `trainer.train()` returns, so a timeout loses everything. |
| `save_strategy="steps"` + `save_steps=0.1` | Save checkpoints for configured persistent storage. |

## Secrets

Secrets are environment variables injected into the Jobs container. They never appear in logs and are not part of the script.

| Secret | Required when |
|---|---|
| `HF_TOKEN` | Only when authorized Hub push or authenticated Trackio requires it. |
| `WANDB_API_KEY` | Using `report_to="wandb"`. |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_TOKEN` | Using MLflow with a remote server. |

The `$HF_TOKEN` syntax in the job config references the value from your local environment at submission time. The literal string `$HF_TOKEN` is replaced with your token's value. Never hardcode tokens in the script itself.

Trackio (the default tracker in this skill) uses `HF_TOKEN` for auth, so no extra secrets are needed. Only switch to the W&B / MLflow rows above if you're using those trackers.

## Timeout

Default is **30 minutes**, which is too short for almost any real training. Set explicitly:

```python
"timeout": "2h"       # 2 hours
"timeout": "90m"      # 90 minutes
"timeout": "1.5h"     # 90 minutes
"timeout": 7200       # seconds, as integer
```

Rule: **estimated training time × 1.3**. The extra buffer covers model loading, dataset caching, checkpoint saving, and Hub push.

On timeout, the container is killed immediately. Only data in configured persistent storage survives. Verify artifacts after the job; timeout is not success evidence.

## Dataset caching

Hugging Face datasets are cached at `~/.cache/huggingface/datasets` by default. That's **inside the container**, which is destroyed after the job. Each Jobs run re-downloads the dataset.

For large datasets (>5 GB), this matters. Options:

- **Persistent `/data` volume** (Jobs feature, check current documentation): set `HF_DATASETS_CACHE=/data/datasets` so caches persist across jobs.
- **Pre-cache locally, push to Hub**: if the dataset is on Hub already, nothing to do. If it's local-only, `dataset.push_to_hub(...)` once so subsequent jobs load from Hub.

## Monitoring a running job

```bash
hf jobs ps [--all]                        # running (or all) jobs
hf jobs inspect <job-id>                  # full config + status
hf jobs logs <job-id> [--follow|--tail N] # tail or stream
hf jobs cancel <job-id>
hf jobs hardware                          # list flavors + hourly rates
```

`hf jobs logs <id> --follow` under `Bash run_in_background` pairs nicely with a `Monitor` watching for the `VERDICT:` line emitted by your training script's verdict block.

MCP equivalents (signatures may vary by server version, so check the actual
tool listing): `hf_jobs("ps")`, `hf_jobs("logs", {"job_id": ...})`,
`hf_jobs("cancel", {"job_id": ...})`.

For recurring runs, `hf jobs scheduled uv run "<cron>" <script> ...`
schedules. `hf jobs scheduled ps/suspend/delete` manages.

## Common failures

### Expected artifact missing after a successful-looking run

The job container is temporary. Check job status and approved persistent storage.

Enable Hub upload only after explicit authorization for the exact destination. Otherwise configure this repository's checkpoint bucket or an approved persistent volume, then verify artifact receipt.

### Tracker not connecting

- **Trackio:** `HF_TOKEN` missing or lacks write permission. Add `"secrets": {"HF_TOKEN": "$HF_TOKEN"}` and make sure the token has write access.
- **W&B:** `WANDB_API_KEY` missing. Add `"secrets": {"HF_TOKEN": "$HF_TOKEN", "WANDB_API_KEY": "$WANDB_API_KEY"}`.

### OOM on first step

Flavor too small. Move up one tier (see `hardware_guide.md`).

### Training starts but eval hangs forever

`eval_strategy="steps"` with no `eval_dataset`. Always provide an eval dataset, or set `eval_strategy="no"`.

### Dataset download times out

Large dataset or slow cold-cache. Increase `timeout` or pre-cache to a persistent volume.

### `CachedMultipleNegativesRankingLoss` + `gradient_checkpointing=True` crash

The cached losses are incompatible with gradient checkpointing. Disable `gradient_checkpointing`.

After submission, the MCP returns a job ID. Monitor with `hf_jobs("logs", {"job_id": ...})` when you want an update. Don't poll in a tight loop. End-to-end submission templates live in `scripts/train_sentence_transformer_example.py` / `scripts/train_cross_encoder_example.py` / `scripts/train_sparse_encoder_example.py`. Wrap the script contents in the inline pattern from §1 above.
