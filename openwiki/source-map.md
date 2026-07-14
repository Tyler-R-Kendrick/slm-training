# Source map

High-signal paths (not exhaustive):

## Root

| Path | Role |
| --- | --- |
| `AGENTS.md` | Cross-agent instructions |
| `RTK.md` | Rust Token Killer usage |
| `README.md` | Human overview + model-card summary |
| `docs/MODEL_CARD.md` | Checkpoint roster + eval |
| `docs/design/` | Specs + measured results |
| `openwiki/` | Agent wiki (this directory) |
| `.agents/skills/` | Canonical skills |
| `.github/workflows/ci.yml` | Lint / pytest |
| `.github/workflows/openwiki-update.yml` | Scheduled OpenWiki PRs |

## `src/slm_training/`

| Path | Role |
| --- | --- |
| `dsl/` | OpenUI codec / schema |
| `models/` | TwoTower, grammar diffusion, tokenizers |
| `harnesses/model_build/` | Train/eval loop + checkpoint bucket |
| `harnesses/train_data/`, `test_data/` | Data harnesses |
| `grammar_fastpath/`, `grammar_backends/` | Constrained decode |
| `evals/`, `quality/`, `preference/`, `rl/` | Eval / rewards / RL-lite |
| `web/` | Playground API |
| `accel/`, `telemetry/`, `cactus/` | Perf / export |

## `scripts/`

Train/eval/matrix CLIs: `train_model`, `evaluate_model`, `run_quality_matrix`, `run_perf_matrix`, `run_grammar_matrix`, `hf_jobs_train`, `remote_train`, `sync_checkpoints`, `bootstrap_playground`, …

## Related packages

| Path | Role |
| --- | --- |
| `src/gpu_multi_farm/` | Multi-farm MCP server |
| `tools/openui_*` | Node bridges / preview |
| `fixtures/` | Seed pairs + RICO slices |
| `api/` | Vercel FastAPI entry |
