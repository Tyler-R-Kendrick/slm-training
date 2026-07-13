# slm-training

Novel SLM experiments: harnesses for **placeholder OpenUI** layout generation (official `@openuidev/lang-core`), plus a **GPU multi-farm MCP**.

## What's included

1. **Training-data harness** — build/validate versioned train corpora
2. **Testing-data harness** — held-out / adversarial / OOD eval suites
3. **Model-building harness** — train/eval shell with a stub plug-in (no TwoTower yet)
4. **OpenUI Lang bridge** — Node sidecar over official `@openuidev/lang-core`
5. **GPU multi-farm MCP** — list / launch / cost-project across Vast.ai, RunPod, Lambda

See [docs/design/openui-twotower.md](docs/design/openui-twotower.md) and [docs/design/gpu-multi-farm-mcp.md](docs/design/gpu-multi-farm-mcp.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Official OpenUI parser bridge (required for harness validate/eval)
cd tools/openui_bridge && npm ci && cd ../..

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

# Official teacher system prompt (for synth / distillation)
python -m scripts.export_openui_prompt
```

```bash
pytest
```

## OpenUI Lang

Fixtures and validation use **real OpenUI Lang** positional syntax, e.g.:

```
root = Stack([hero], "vertical")
hero = Card(":hero.title", ":hero.body")
```

Content props must be placeholder strings. Parsing/serialization/prompt generation come from `@openuidev/lang-core` — see [`tools/openui_bridge/`](tools/openui_bridge/).

## GPU multi-farm MCP

```bash
cp .env.example .env
pip install -e ".[mcp]"
GPU_MULTI_FARM_MODE=mock python -m scripts.multi_farm_mcp
```

## Layout

```
src/slm_training/dsl/           # Python adapter + record schema
src/slm_training/harnesses/     # train_data, test_data, model_build
src/gpu_multi_farm/             # FastMCP server + farm adapters
tools/openui_bridge/            # @openuidev/lang-core Node sidecar
scripts/                        # CLIs
fixtures/                       # seed pairs (OpenUI Lang)
docs/design/                    # architecture + contracts
```
