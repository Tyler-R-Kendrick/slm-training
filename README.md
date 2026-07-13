# slm-training

Novel SLM experiments: harnesses for **placeholder OpenUI** layout generation (official `@openuidev/lang-core`), a **TwoTower** masked-diffusion model, plus a **GPU multi-farm MCP**.

## What's included

1. **Training-data harness** — build/validate versioned train corpora
2. **Testing-data harness** — held-out / adversarial / OOD eval suites
3. **Model-building harness** — train/eval with **TwoTower** (default) or stub plug-in
4. **OpenUI Lang bridge** — Node sidecar over official `@openuidev/lang-core`
5. **GPU multi-farm MCP** — list / launch / cost-project across Vast.ai, RunPod, Lambda

See [docs/design/openui-twotower.md](docs/design/openui-twotower.md) and [docs/design/gpu-multi-farm-mcp.md](docs/design/gpu-multi-farm-mcp.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torch]"

# Official OpenUI parser + DESIGN.md bridges
cd tools/openui_bridge && npm ci && cd ../..
cd tools/design_md_bridge && npm ci && cd ../..

# optional MCP server deps
pip install -e ".[mcp]"
# optional live RICO download
pip install -e ".[rico]"
```

## Quick start (train / disjoint test)

```bash
# High-quality versioned corpus (default: all sources + quality synthesizer)
python -m scripts.build_train_data --source all --version v1 --synthesizer quality

# Fast fixture-only rebuild
python -m scripts.build_train_data --source fixture --version v0 --synthesizer quality

# Test suites with strict leakage checks against the train manifest
python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/train_data/v1/manifest.json

python -m scripts.train_model \
  --train-dir outputs/train_data/v1 \
  --model twotower \
  --context-backend hf \
  --steps 200 \
  --run-id twotower_v1

python -m scripts.evaluate_model \
  --test-dir outputs/test_data/v1 \
  --suite smoke \
  --model twotower \
  --run-id twotower_v1 \
  --fail-under-parse-rate 0.5
```

Train artifacts land in `outputs/train_data/<version>/` (`train.jsonl`, `val.jsonl`, `test.jsonl`, `manifest.json`, `stats.json`). The flush pipeline: curated seeds + RICO + Awwwards → deterministic quality synth (layout augment + prompt paraphrase) → per-record DESIGN.md + OpenUI validate → quality gates → stable sort by `id` + content fingerprint.

```bash
pytest
```

## OpenUI Lang

Fixtures and validation use official **`openuiLibrary`** syntax, e.g.:

```
root = Stack([hero], "column")
hero_title = TextContent(":hero.title")
hero_body = TextContent(":hero.body")
hero = Card([hero_title, hero_body])
```

Content props must be placeholder strings. Parsing/serialization/prompt generation come from `@openuidev/lang-core` + `@openuidev/react-ui` — see [`tools/openui_bridge/`](tools/openui_bridge/).

DESIGN.md conditioning + linter: [`tools/design_md_bridge/`](tools/design_md_bridge/) and [`fixtures/design_md/`](fixtures/design_md/).

## Web playground

```bash
pip install -e ".[dev,torch,web]"
python -m scripts.bootstrap_playground   # trains a tiny demo checkpoint
python -m scripts.serve_playground --port 8765
# open http://127.0.0.1:8765
```


- **Context tower**: scratch TokenEncoder **or** frozen HF model (`--context-backend hf`, default `HuggingFaceTB/SmolLM2-135M`)
- **Denoiser tower**: MaskGIT-style masked token prediction with cross-attention to context
- **Grammar decode**: official `createStreamingParser` guards unmasking / LTR repair (`--no-grammar` to disable)
- **Tokenizer**: OpenUI-aware whitespace/placeholder tokenizer built from train artifacts
- **Eval**: `parse_rate` via lang-core, placeholder fidelity, canonical tree match — no gold oracle at generate time

```bash
# Optional HF context (requires: pip install -e ".[hf]")
python -m scripts.train_model --model twotower --context-backend hf \
  --hf-model HuggingFaceTB/SmolLM2-135M --steps 200 --run-id twotower_hf
```

## GPU multi-farm MCP

```bash
cp .env.example .env
pip install -e ".[mcp]"
GPU_MULTI_FARM_MODE=mock python -m scripts.multi_farm_mcp
```

## Layout

```
src/slm_training/dsl/           # Python adapter + record schema
src/slm_training/data/          # RICO adapters + leakage fingerprints
src/slm_training/models/        # TwoTower + OpenUI tokenizer
src/slm_training/harnesses/     # train_data, test_data, model_build
src/gpu_multi_farm/             # FastMCP server + farm adapters
tools/openui_bridge/            # @openuidev/lang-core Node sidecar
scripts/                        # CLIs
fixtures/                       # seed pairs + RICO semantic slices
docs/design/                    # architecture + contracts
```
