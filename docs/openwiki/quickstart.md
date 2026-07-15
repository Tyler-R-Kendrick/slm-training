# slm-training quickstart

Experiment-first repository for **OpenUI layout SLMs** (TwoTower / grammar-diffusion): honest multi-suite ship gates, durable `docs/design/` evidence, HF checkpoint bucket sync, and agent-facing instructions.

## What this repo does

- Trains and evaluates structure-aware layout models (`src/slm_training/`).
- Runs quality / grammar / perf experiment matrices via `scripts/`.
- Records measured results under `docs/design/` (JSON + markdown) and keeps `docs/MODEL_CARD.md` in sync with checkpoints.
- Syncs full HF-context trains to [OpenUI HF Bucket](https://huggingface.co/buckets/TKendrick/OpenUI).
- Hosts agent skills under `.agents/skills/` (Claude / Cursor / Codex / GHCP).

## Start here

| Need | Go to |
| --- | --- |
| Cross-agent process rules | [`AGENTS.md`](../../AGENTS.md) |
| Architecture | [Architecture overview](./architecture/overview.md) |
| Train / eval / docs loop | [Train → eval → docs](./workflows/train-eval-docs.md) |
| Checkpoints + model cards | [Checkpoints & agents](./operations/checkpoints-and-agents.md) |
| Tests / ship gates | [Testing guidance](./testing/guidance.md) |
| File placement / moves | [`docs/repository-organization.md`](../repository-organization.md) |
| Playground / MCP farms | [Integrations](./integrations/surfaces.md) |
| Source map | [Source map](./source-map.md) |

Canonical design specs (read next when implementing):

- [`docs/design/openui-twotower.md`](../design/openui-twotower.md)
- [`docs/design/research-lineage.md`](../design/research-lineage.md)
- [`docs/design/quality-experiment-matrix.md`](../design/quality-experiment-matrix.md)
- [`docs/design/adversarial-review.md`](../design/adversarial-review.md)
- [`docs/design/checkpoint-bucket.md`](../design/checkpoint-bucket.md)
- [`docs/MODEL_CARD.md`](../MODEL_CARD.md)

## Fixture vs ship (confirmed)

- Scratch / quality-matrix / CI fixture demos are **wiring only**.
- Production / ship claims need full scoreboards (`rico_held` / HF / DESIGN.md as claimed) via `honest-ship-eval`.
- Full HF trains without a successful `checkpoint_bucket` remote URI (or explicit `--no-sync-checkpoints`) are incomplete.

## OpenWiki refresh

```bash
npm install -g openwiki@0.1.2
python -m scripts.update_openwiki --init      # first generation (needs provider API key)
python -m scripts.update_openwiki --update --print
```

Scheduled refresh: [`.github/workflows/openwiki-update.yml`](../../.github/workflows/openwiki-update.yml) (prefers `OPENAI_API_KEY`, then `OPENROUTER_API_KEY`; optional `LANGSMITH_API_KEY` enables tracing).

## Backlog

- Run and review the scheduled OpenWiki refresh once a provider secret is configured.
- Domain pages for grammar-fastpath / speculative denoising when those surfaces change.
