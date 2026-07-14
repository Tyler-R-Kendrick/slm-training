A code wiki for the **slm-training** OpenUI layout-SLM research repo.

Prioritize:

1. Concise quickstart for agents and engineers (where to start, what is fixture vs ship).
2. Architecture: TwoTower / grammar-diffusion, harnesses (`train_data` / `test_data` / `model_build`), DSL + grammar backends.
3. Core workflows: train → eval → matrix → docs/MODEL_CARD updates; HF checkpoint bucket sync.
4. Honesty gates (`honest-ship-eval`) vs scratch / quality-matrix demos.
5. Agent surface: `AGENTS.md`, `.agents/skills/`, token-efficiency stack, OpenWiki itself.
6. Source map: `src/slm_training/`, `scripts/`, `docs/design/`, playground / Vercel entrypoints.

Ground pages in repository structure and measured results under `docs/design/`. Prefer practical navigation over generic summaries. Prefer linking existing design docs over rewriting them.

Keep generated pages under `openwiki/`; do not rewrite this `INSTRUCTIONS.md` during routine `--update` runs unless the user asks.
