# Integrations & product surfaces

## Vercel / playground

- FastAPI entry: `api/index.py` (`pyproject.toml` `[tool.vercel]`)
- Web package: `src/slm_training/web/`
- Bootstrap: `scripts/bootstrap_playground.py`

## Hugging Face

- CLI + marketplace skills: `.agents/skills/hf-*`, `huggingface-*`
- MCP: `.cursor/mcp.json` → `https://huggingface.co/mcp?login`
- Checkpoint bucket sync for full trains (see operations page)

## GPU multi-farm MCP

- Package: `src/gpu_multi_farm/`
- Script: `scripts/multi_farm_mcp.py`
- Design: [`docs/design/gpu-multi-farm-mcp.md`](../docs/design/gpu-multi-farm-mcp.md)
- Env: `VAST_API_KEY` / `RUNPOD_API_KEY` / `LAMBDA_API_KEY` in `.env` (never commit)

## OpenUI bridge tooling

- `tools/openui_bridge/`, `tools/design_md_bridge/`, `tools/openui_preview/`
- Node sidecars for `@openuidev/lang-core` / DESIGN.md
