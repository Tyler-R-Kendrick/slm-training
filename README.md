# slm-training

Novel SLM experiments: harnesses for **placeholder OpenUI** layout generation, plus a **GPU multi-farm MCP** for cheap training pods.

## What's included

1. **Training-data harness** — build/validate versioned train corpora
2. **Testing-data harness** — held-out / adversarial / OOD eval suites
3. **Model-building harness** — train/eval shell with a stub plug-in (no TwoTower yet)
4. **GPU multi-farm MCP** — list / launch / cost-project across Vast.ai, RunPod, Lambda

See [docs/design/openui-twotower.md](docs/design/openui-twotower.md) and [docs/design/gpu-multi-farm-mcp.md](docs/design/gpu-multi-farm-mcp.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# optional MCP server deps
pip install -e ".[mcp]"
```

## Quick start (offline harnesses)

```bash
python -m scripts.build_train_data --version v0
python -m scripts.build_test_data --version v0 \
  --train-manifest outputs/train_data/v0/manifest.json
python -m scripts.train_model --train-dir outputs/train_data/v0 --steps 2
python -m scripts.evaluate_model \
  --test-dir outputs/test_data/v0 \
  --suite smoke \
  --checkpoint outputs/runs/latest/checkpoints/last.pt
```

```bash
pytest
```

## GPU multi-farm MCP

```bash
cp .env.example .env   # set VAST_API_KEY / RUNPOD_API_KEY / LAMBDA_API_KEY as needed
pip install -e ".[mcp]"
GPU_MULTI_FARM_MODE=mock python -m scripts.multi_farm_mcp
```

Tools: `list_available_gpus`, `launch_training_pod`, `project_training_cost`.

Cursor MCP config example:

```json
{
  "mcpServers": {
    "gpu-multi-farm": {
      "command": "python",
      "args": ["-m", "scripts.multi_farm_mcp"],
      "cwd": "/absolute/path/to/slm-training",
      "env": { "GPU_MULTI_FARM_MODE": "auto" }
    }
  }
}
```

## Layout

```
src/slm_training/dsl/           # shared grammar / schema
src/slm_training/harnesses/     # train_data, test_data, model_build
src/gpu_multi_farm/             # FastMCP server + farm adapters
scripts/                        # CLIs + multi_farm_mcp entrypoint
fixtures/                       # seed pairs for offline CI
docs/design/                    # architecture + contracts
```
