# Operations: checkpoints & agents

## Checkpoint bucket

- **URI:** `hf://buckets/TKendrick/OpenUI`
- **Web:** https://huggingface.co/buckets/TKendrick/OpenUI
- **Layout:** `checkpoints/<run_id>/`
- **Auth:** `HF_TOKEN` (never commit)

| Entry | Sync default |
| --- | --- |
| `scripts.train_model` (`--context-backend hf`) | on |
| `scripts.remote_train` | on |
| `scripts.hf_jobs_train` | on |
| Programmatic `ModelBuildConfig` / pytest | off |
| Quality matrix / scratch | off |

Disable: `--no-sync-checkpoints`, `SLM_DISABLE_CHECKPOINT_BUCKET=1`, or empty `checkpoint_bucket`.

Details: [`docs/design/checkpoint-bucket.md`](../../design/checkpoint-bucket.md).

## Model card

Every new/promoted checkpoint must refresh:

1. [`docs/MODEL_CARD.md`](../../MODEL_CARD.md)
2. README “Model card (summary)”

## Agent discovery

| Path | Audience |
| --- | --- |
| `AGENTS.md` | All coding agents (canonical) |
| `CLAUDE.md` / `GEMINI.md` | Thin pointers |
| `.agents/skills/` | Skills (Codex / Cursor / GHCP project path) |
| `.claude/skills/`, `.cursor/skills/` | Symlinked discovery |
| `docs/repository-organization.md` | Canonical placement, deduplication, and move policy |
| `.github/copilot-instructions.md` | GHCP Chat |
| `docs/openwiki/` | This wiki (agent context via OpenWiki snippets) |

## OpenWiki code mode

- Workflow: `.github/workflows/openwiki-update.yml`
- Control brief: `docs/openwiki/INSTRUCTIONS.md` (do not regenerate casually)
- Refresh: `python -m scripts.update_openwiki --update --print`; CI prefers `OPENAI_API_KEY`, then `OPENROUTER_API_KEY`, with optional LangSmith tracing.
