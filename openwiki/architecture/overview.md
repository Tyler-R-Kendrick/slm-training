# Architecture overview

## Layers

| Layer | Location | Role |
| --- | --- | --- |
| DSL / codec | `src/slm_training/dsl/` | OpenUI adapter, schema, parser, production codec |
| Models | `src/slm_training/models/` | TwoTower, grammar_diffusion, tokenizers, remask |
| Harnesses | `src/slm_training/harnesses/` | `train_data`, `test_data`, `model_build` (train/eval loops) |
| Grammar | `src/slm_training/grammar_fastpath/`, `grammar_backends/` | DFA force-emit, MaskGIT admit, lang-core / Lark / hybrid |
| Agents / skills | `AGENTS.md`, `.agents/skills/` | Cross-tool process + skills |
| Design evidence | `docs/design/` | Matrices, measured results, research lineage |
| Serving / web | `src/slm_training/web/`, `api/` | Annotate playground API; Vercel FastAPI entry |

## Model build path (confirmed)

`ModelBuildConfig` drives train/eval. Checkpoint sync defaults **off** for programmatic/pytest harness calls; `scripts.train_model` enables sync for HF-context full trains (`docs/design/checkpoint-bucket.md`).

Key modules:

- `src/slm_training/harnesses/model_build/train_loop.py` — training loop + post-train bucket sync hook
- `src/slm_training/harnesses/model_build/checkpoint_bucket.py` — HF bucket URI helpers
- `scripts/train_model.py` / `scripts/remote_train.py` / `scripts/sync_checkpoints.py`

## Research → code map

See [`docs/design/research-lineage.md`](../docs/design/research-lineage.md) for paper → implementation pointers (remask, grammar fastpath, speculative denoising, verifier-guided repair).

## Extension points

- New matrix cells: `docs/design/quality-experiment-matrix.md` / `perf-experiment-matrix.md` + `scripts/run_*_matrix.py`
- New skill: `.agents/skills/<name>/SKILL.md` (mirror/symlink to `.claude` / `.cursor`)
- GPU farm offers: `src/gpu_multi_farm/` + MCP (`docs/design/gpu-multi-farm-mcp.md`)
