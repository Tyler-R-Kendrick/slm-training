# Integrations & product surfaces

## Vercel / playground

- FastAPI entry: `src/slm_training/web/vercel.py` (`pyproject.toml` `[tool.vercel]`)
- Web package: `src/slm_training/web/`
- Bootstrap: `scripts/bootstrap_playground.py`

## Hugging Face

- CLI + marketplace skills: `.agents/skills/hf-*`, `huggingface-*`
- MCP: `.cursor/mcp.json` → `https://huggingface.co/mcp?login`
- Checkpoint bucket sync for full trains (see operations page)

## Serena MCP

- Semantic IDE tools for agents ([oraios/serena](https://github.com/oraios/serena))
- Project: `.serena/project.yml` (languages: python, typescript)
- Clients: `.cursor/mcp.json`, `.mcp.json` (Claude Code), `.vscode/mcp.json`
- Prefer `find_symbol` / `find_referencing_symbols` over raw grep when editing Python/TS

## GPU multi-farm MCP

- Package: `src/gpu_multi_farm/`
- Script: `scripts/multi_farm_mcp.py`
- Design: [`docs/design/gpu-multi-farm-mcp.md`](../../design/gpu-multi-farm-mcp.md)
- Env: `VAST_API_KEY` / `RUNPOD_API_KEY` / `LAMBDA_API_KEY` in `.env` (never commit)

## OpenUI bridge tooling

- `src/apps/openui_bridge/`, `src/apps/design_md_bridge/`, `src/apps/openui_preview/`
- Node sidecars for `@openuidev/lang-core` / DESIGN.md
